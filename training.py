"""Durable, unlabeled Label Studio capture queue. No classifier dependency."""
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
import uuid

from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError
from labelstudio import LabelStudioUploader, SETUP, UploadError, valid_url

MAX_IMAGES = 500
MAX_JPEG = 2 * 1024 * 1024


class CaptureQueue:
    def __init__(self, directory, key="", url="", project="", uploader=None, public_url=""):
        self.key, self.url, self.project = key, url.rstrip("/"), str(project)
        self.public_url = (public_url or url).rstrip("/")
        self.configured = bool(key and valid_url(self.url) and valid_url(self.public_url)
                               and re.fullmatch(r"[1-9][0-9]*", self.project))
        Path(directory).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = str(Path(directory) / "captures.sqlite3")
        self.uploader = uploader or LabelStudioUploader(self.url, self.key).upload
        self.wake = threading.Event()
        self.closed = threading.Event()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY, session TEXT NOT NULL, workspace TEXT NOT NULL,
                project TEXT NOT NULL, digest TEXT NOT NULL, jpeg BLOB,
                state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, lease TEXT, error TEXT,
                roboflow_id TEXT, created REAL NOT NULL)""")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(captures)")}
            for name, kind in (("provider", "TEXT"), ("base_url", "TEXT"), ("task_id", "INTEGER")):
                if name not in columns:
                    db.execute(f"ALTER TABLE captures ADD COLUMN {name} {kind}")
            # Leave completed Roboflow records intact; their JPEGs were already removed.
            db.execute("UPDATE captures SET provider='roboflow' WHERE provider IS NULL AND state='uploaded'")
            if self.configured:
                db.execute("""UPDATE captures SET provider='labelstudio', base_url=?, project=?,
                    state='pending', next_attempt=0, lease=NULL, error=NULL
                    WHERE provider IS NULL AND state != 'uploaded'""", (self.url, self.project))
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
                    project_url=f"{self.public_url}/projects/{self.project}/data/" if self.configured else None)

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
            db.execute("INSERT INTO captures (id,session,workspace,project,digest,jpeg,created,provider,base_url) VALUES (?,?,?,?,?,?,?,?,?)",
                       (capture_id, session, "", self.project, digest, jpeg, time.time(), "labelstudio", self.url))
        self.wake.set()
        return {"id": capture_id, "state": "pending", "duplicate": False}, 201

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
            if row["base_url"] != self.url or row["project"] != self.project:
                raise UploadError("The queued Label Studio destination differs from current settings. Retry uploads to use the current project.", True)
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
                db.execute("UPDATE captures SET state='uploaded', jpeg=NULL, error=NULL, task_id=?, lease=NULL WHERE id=? AND lease=?", (image_id, row["id"], lease))
        return True

    def retry(self):
        with self.connect() as db:
            # Explicit retry applies corrected configuration to rejected records.
            db.execute("UPDATE captures SET base_url=?, project=?, provider='labelstudio' WHERE state='failed'",
                       (self.url, self.project))
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
        os.environ.get("LABEL_STUDIO_API_KEY", ""), os.environ.get("LABEL_STUDIO_URL", ""),
        os.environ.get("LABEL_STUDIO_PROJECT_ID", ""), public_url=os.environ.get("LABEL_STUDIO_PUBLIC_URL", ""))
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
