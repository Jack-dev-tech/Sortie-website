"""Durable, unlabeled Roboflow capture queue. No classifier dependency."""
from contextlib import contextmanager
import base64
import binascii
import hashlib
import io
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import quote
import uuid

from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError
import requests

MAX_IMAGES = 500
MAX_JPEG = 2 * 1024 * 1024
SETUP = "Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT on the server, then restart."


class UploadError(Exception):
    def __init__(self, message, permanent=False):
        super().__init__(message)
        self.permanent = permanent


class CaptureQueue:
    def __init__(self, directory, key="", workspace="", project="", uploader=None):
        self.key, self.workspace, self.project = key, workspace, project
        self.configured = bool(key and re.fullmatch(r"[\w-]+", workspace) and re.fullmatch(r"[\w-]+", project))
        Path(directory).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = str(Path(directory) / "captures.sqlite3")
        self.uploader = uploader or self.upload
        self.wake = threading.Event()
        self.closed = threading.Event()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY, session TEXT NOT NULL, workspace TEXT NOT NULL,
                project TEXT NOT NULL, digest TEXT NOT NULL, jpeg BLOB,
                state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, lease TEXT, error TEXT,
                roboflow_id TEXT, created REAL NOT NULL)""")
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def status(self):
        with self.connect() as db:
            counts = dict(db.execute("SELECT state, COUNT(*) FROM captures GROUP BY state").fetchall())
            error = db.execute("SELECT error FROM captures WHERE error IS NOT NULL ORDER BY created DESC LIMIT 1").fetchone()
        pending = sum(v for k, v in counts.items() if k != "uploaded")
        failed = counts.get("failed", 0)
        return dict(configured=self.configured, captured=sum(counts.values()),
                    uploaded=counts.get("uploaded", 0), pending=pending, failed=failed,
                    capacity=MAX_IMAGES, full=pending >= MAX_IMAGES,
                    can_capture=self.configured and pending < MAX_IMAGES and not failed,
                    error=(error[0] if error else None),
                    setup=None if self.configured else SETUP,
                    project_url=f"https://app.roboflow.com/{quote(self.workspace)}/{quote(self.project)}" if self.configured else None)

    def save(self, capture_id, session, jpeg):
        digest = hashlib.sha256(jpeg).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT digest, session, state FROM captures WHERE id=?", (capture_id,)).fetchone()
            if existing:
                if existing["digest"] != digest or existing["session"] != session:
                    return {"error": "Capture ID already belongs to a different image or session."}, 409
                return {"id": capture_id, "state": existing["state"], "duplicate": True}, 200
            if not self.configured:
                return {"error": SETUP}, 503
            if db.execute("SELECT COUNT(*) FROM captures WHERE state != 'uploaded'").fetchone()[0] >= MAX_IMAGES:
                return {"error": "Queue full (500 images). Capture paused until uploads make room."}, 507
            if db.execute("SELECT 1 FROM captures WHERE state='failed' LIMIT 1").fetchone():
                return {"error": "Upload needs attention. Resolve the upload error, then retry uploads."}, 503
            db.execute("INSERT INTO captures (id,session,workspace,project,digest,jpeg,created) VALUES (?,?,?,?,?,?,?)",
                       (capture_id, session, self.workspace, self.project, digest, jpeg, time.time()))
        self.wake.set()
        return {"id": capture_id, "state": "pending", "duplicate": False}, 201

    def upload(self, row):
        # Validate the configured workspace/project before uploading. Never return or log
        # upstream responses/exceptions: either can contain a credential-bearing URL.
        try:
            info = requests.get(f"https://api.roboflow.com/{row['workspace']}/{row['project']}",
                                params={"api_key": self.key}, timeout=(10, 30))
            self.check_response(info)
            project = info.json().get("project", {})
            if project.get("type") != "object-detection":
                raise UploadError("Use a Roboflow object-detection project; check workspace/project settings and restart, then retry uploads.", True)
            response = requests.post(f"https://api.roboflow.com/dataset/{row['project']}/upload",
                params={"api_key": self.key, "batch": f"sortie-{row['session']}"},
                data={"name": f"{row['id']}.jpg", "split": "train"},
                files={"file": (f"{row['id']}.jpg", row["jpeg"], "image/jpeg")}, timeout=(10, 45))
            self.check_response(response)
            data = response.json()
            if not (data.get("success") or data.get("duplicate")) or not isinstance(data.get("id"), str) or not data["id"]:
                raise UploadError("Roboflow did not confirm an image ID. Check project access, then retry uploads.", True)
            return data["id"]
        except (requests.RequestException, ValueError):
            raise UploadError("Roboflow is unreachable or returned an invalid response. Retrying automatically.") from None

    @staticmethod
    def check_response(response):
        if response.status_code in (401, 403):
            raise UploadError("Roboflow authentication failed. Check the server API key and project permissions, restart, then retry uploads.", True)
        if response.status_code == 429 or response.status_code >= 500 or response.status_code == 408:
            raise UploadError("Roboflow is temporarily unavailable or rate limited. Retrying automatically.")
        if not response.ok:
            raise UploadError("Roboflow rejected the upload. Check workspace/project settings and access, then retry uploads.", True)

    def process_one(self):
        if not self.configured:
            return False
        now, lease = time.time(), str(uuid.uuid4())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A lease survives process death and avoids duplicate workers claiming a row.
            row = db.execute("SELECT * FROM captures WHERE state IN ('pending','uploading') AND next_attempt<=? ORDER BY created LIMIT 1", (now,)).fetchone()
            if row is None:
                return False
            db.execute("UPDATE captures SET state='uploading', lease=?, next_attempt=? WHERE id=?", (lease, now + 180, row["id"]))
        try:
            image_id = self.uploader(row)
        except Exception as exc:
            known = isinstance(exc, UploadError)
            permanent = known and exc.permanent
            message = str(exc) if known else "Upload interrupted. Retrying automatically."
            delay = min(300, 3 * 2 ** min(row["attempts"], 7))
            with self.connect() as db:
                db.execute("UPDATE captures SET state=?, attempts=attempts+1, next_attempt=?, error=?, lease=NULL WHERE id=? AND lease=?",
                           ("failed" if permanent else "pending", time.time() + delay, message, row["id"], lease))
        else:
            with self.connect() as db:
                db.execute("UPDATE captures SET state='uploaded', jpeg=NULL, error=NULL, roboflow_id=?, lease=NULL WHERE id=? AND lease=?", (image_id, row["id"], lease))
        return True

    def retry(self):
        with self.connect() as db:
            # Explicit retry applies corrected configuration to rejected records.
            db.execute("UPDATE captures SET workspace=?, project=? WHERE state='failed'",
                       (self.workspace, self.project))
            db.execute("UPDATE captures SET state='pending', next_attempt=0, error=NULL WHERE state IN ('pending','failed')")
        self.wake.set()

    def start(self):
        def run():
            while not self.closed.is_set():
                try:
                    worked = self.process_one()
                except sqlite3.Error:
                    worked = False
                if not worked:
                    self.wake.wait(2)
                    self.wake.clear()
        threading.Thread(target=run, name="training-uploader", daemon=True).start()


def install_training(app, queue=None, start_worker=True):
    queue = queue or CaptureQueue(os.environ.get("SORTIE_TRAINING_DIR", str(Path(app.root_path) / "data" / "training")),
        os.environ.get("ROBOFLOW_API_KEY", ""), os.environ.get("ROBOFLOW_WORKSPACE", ""), os.environ.get("ROBOFLOW_PROJECT", ""))
    app.extensions["training_queue"] = queue
    routes = Blueprint("training", __name__)

    @routes.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @routes.get("/training/status")
    def status():
        return jsonify(queue.status())

    @routes.post("/training/retry")
    def retry():
        if not queue.configured:
            return jsonify(error=SETUP), 503
        queue.retry()
        return jsonify(queue.status())

    @routes.post("/training/captures")
    def capture():
        if request.content_length and request.content_length > MAX_JPEG * 2:
            return jsonify(error="Image is too large."), 413
        request.max_content_length = MAX_JPEG * 2
        data = request.get_json(silent=True)
        try:
            if not isinstance(data, dict):
                raise ValueError()
            capture_id, session = (str(uuid.UUID(data[k])) for k in ("id", "session"))
            image = data["image"]
            if not isinstance(image, str) or not image.startswith("data:image/jpeg;base64,"):
                raise ValueError()
            jpeg = base64.b64decode(image.split(",", 1)[1], validate=True)
            if not jpeg or len(jpeg) > MAX_JPEG:
                raise ValueError()
            with Image.open(io.BytesIO(jpeg)) as im:
                if im.format != "JPEG" or not (1 <= im.width <= 640 and 1 <= im.height <= 1920):
                    raise ValueError()
                im.verify()
            with Image.open(io.BytesIO(jpeg)) as im:
                im.load()
        except (KeyError, TypeError, AttributeError, ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="Provide UUID id/session and a valid JPEG up to 640 pixels wide and 2 MB."), 400
        try:
            result, code = queue.save(capture_id, session, jpeg)
            return jsonify(result), code
        except sqlite3.Error:
            return jsonify(error="Local capture storage is unavailable. Check disk space and permissions."), 503

    app.register_blueprint(routes)
    if start_worker:
        queue.start()
    return queue
