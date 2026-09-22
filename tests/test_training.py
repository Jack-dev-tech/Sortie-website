import base64
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
import uuid

from flask import Flask
from PIL import Image
import httpx
import jwt
import json
import sqlite3
from label_studio_sdk import LabelStudio
from labelstudio import LabelStudioUploader, valid_url
from training import CaptureQueue, UploadError, install_training


class QueueFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.q = CaptureQueue(self.tmp.name, 'secret-test-key', 'http://localhost:8080', '1', uploader=lambda row: 42)
        self.app = Flask(__name__)
        install_training(self.app, self.q, start_worker=False)
        self.client = self.app.test_client()
        out = io.BytesIO()
        Image.new('RGB', (640, 360), 'red').save(out, format='JPEG', quality=95)
        self.jpeg = out.getvalue()
        self.payload = dict(id=str(uuid.uuid4()), session=str(uuid.uuid4()), image='data:image/jpeg;base64,' + base64.b64encode(self.jpeg).decode())

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        return self.client.post('/training/captures', json=self.payload)


class TrainingTests(QueueFixture):
    def test_success_dedupe_and_jpeg_removal(self):
        self.assertEqual(self.save().status_code, 201)
        self.assertEqual(self.save().status_code, 200)
        self.assertTrue(self.q.process_one())
        self.assertEqual(self.save().json['state'], 'uploaded')
        with self.q.connect() as db:
            row = db.execute('SELECT * FROM captures').fetchone()
            self.assertIsNone(row['jpeg'])
            self.assertEqual(row['task_id'], 42)
        self.assertEqual(self.q.status()['uploaded'], 1)
        self.assertEqual(self.q.status()['pending'], 0)
        self.payload['session'] = str(uuid.uuid4())
        self.assertEqual(self.save().status_code, 409)

    def test_invalid_inputs(self):
        for payload in [[], None, {}, {**self.payload, 'id': '../bad'}, {**self.payload, 'image': 12},
                        {**self.payload, 'image': 'data:image/jpeg;base64,YmFk'},
                        {**self.payload, 'image': 'data:image/jpeg;base64,@@@'}]:
            self.assertEqual(self.client.post('/training/captures', json=payload).status_code, 400)
        for size, fmt in [((641, 360), 'JPEG'), ((100, 100), 'PNG')]:
            out = io.BytesIO(); Image.new('RGB', size).save(out, format=fmt)
            self.payload['image'] = 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()
            self.assertEqual(self.save().status_code, 400)
        self.assertEqual(self.client.post('/training/captures', data=b'x' * (4 * 1024 * 1024 + 1)).status_code, 413)

    def test_unconfigured(self):
        self.q.configured = False
        self.assertEqual(self.save().status_code, 503)
        self.assertFalse(self.q.process_one())
        self.assertFalse(self.q.status()['can_capture'])
        self.assertNotIn('secret-test-key', str(self.client.get('/training/status').json))

    def test_outage_backoff_and_restart(self):
        self.save()
        self.q.uploader = Mock(side_effect=UploadError('Temporary outage'))
        self.q.process_one()
        self.assertFalse(self.q.process_one())
        self.assertEqual(self.q.status()['pending'], 1)
        recovered = CaptureQueue(self.tmp.name, 'key', 'http://localhost:8080', '1', uploader=lambda row: 43)
        recovered.retry()
        self.assertTrue(recovered.process_one())
        self.assertEqual(recovered.status()['uploaded'], 1)

    def test_interrupted_worker_lease_recovery(self):
        self.save()
        with self.q.connect() as db:
            db.execute("UPDATE captures SET state='uploading', lease='old', next_attempt=?", (time.time() + 180,))
        self.assertFalse(self.q.process_one())
        with self.q.connect() as db:
            db.execute('UPDATE captures SET next_attempt=0')
        self.assertTrue(self.q.process_one())
        self.assertEqual(self.q.status()['uploaded'], 1)

    def test_permanent_failure_manual_retry(self):
        self.save()
        self.q.uploader = Mock(side_effect=UploadError('Check API key', True))
        self.q.process_one()
        self.assertFalse(self.q.status()['can_capture'])
        self.assertFalse(self.q.process_one())
        self.q.project = '2'
        self.assertEqual(self.client.post('/training/retry').status_code, 200)
        with self.q.connect() as db:
            self.assertEqual(db.execute('SELECT project FROM captures').fetchone()[0], '2')
        self.q.uploader = lambda row: 44
        self.q.process_one()
        self.assertTrue(self.q.status()['can_capture'])

    def test_capacity_atomic_under_concurrent_submissions(self):
        # Small limit exercises the same transaction as the production 500 limit.
        with patch('training.MAX_IMAGES', 3), ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(lambda _: self.q.save(str(uuid.uuid4()), self.payload['session'], self.jpeg)[1], range(10)))
            self.assertEqual(codes.count(201), 3)
            self.assertEqual(codes.count(507), 7)
            self.assertTrue(self.q.status()['full'])
            self.q.process_one()
            self.assertFalse(self.q.status()['full'])

    def test_public_url_and_invalid_settings(self):
        self.q.public_url = 'https://sortie.example.ts.net'
        self.assertEqual(self.q.status()['project_url'], 'https://sortie.example.ts.net/projects/1/data/')
        for url in ['javascript:alert(1)', 'http://key@host', 'https://host?token=secret',
                    'https://host/#secret', 'https://host/path', 'http://host:bad', 'https://bad host']:
            self.assertFalse(valid_url(url), url)
        for project in ['', 'abc', '-1', '0', '1/other']:
            q = CaptureQueue(self.tmp.name, 'key', 'http://localhost:8080', project)
            self.assertFalse(q.configured)

    def test_changed_destination_requires_explicit_retry(self):
        self.save()
        self.q.url = 'https://new.example.ts.net'
        self.q.uploader = Mock(return_value=42)
        self.q.process_one()
        self.q.uploader.assert_not_called()
        self.assertEqual(self.q.status()['failed'], 1)
        self.q.retry()
        self.q.process_one()
        self.assertEqual(self.q.status()['uploaded'], 1)

    def test_legacy_migration_and_repeated_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'captures.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute("""CREATE TABLE captures (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, workspace TEXT NOT NULL,
                    project TEXT NOT NULL, digest TEXT NOT NULL, jpeg BLOB,
                    state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, lease TEXT, error TEXT,
                    roboflow_id TEXT, created REAL NOT NULL)""")
                for state in ('pending', 'failed', 'uploading', 'uploaded'):
                    db.execute("""INSERT INTO captures
                        (id,session,workspace,project,digest,jpeg,state,roboflow_id,created)
                        VALUES (?,?,?,?,?,?,?,?,?)""", (state, 'session', 'old-workspace',
                        'old-project', 'digest', None if state == 'uploaded' else self.jpeg,
                        state, 'old-id' if state == 'uploaded' else None, 1))
            # Unconfigured startup must not discard or strand the old queue.
            CaptureQueue(directory)
            for _ in range(2):
                q = CaptureQueue(directory, 'key', 'http://localhost:8080', '1')
                with q.connect() as db:
                    rows = {r['id']: r for r in db.execute('SELECT * FROM captures')}
                self.assertEqual(rows['uploaded']['roboflow_id'], 'old-id')
                self.assertEqual(rows['uploaded']['project'], 'old-project')
                self.assertEqual(rows['uploaded']['provider'], 'roboflow')
                for state in ('pending', 'failed', 'uploading'):
                    self.assertEqual(rows[state]['state'], 'pending')
                    self.assertEqual(rows[state]['jpeg'], self.jpeg)
                    self.assertEqual(rows[state]['project'], '1')
                    self.assertEqual(rows[state]['base_url'], 'http://localhost:8080')


CONFIG = (Path(__file__).resolve().parents[1] / 'deploy/label-studio/label-config.xml').read_text()


class LabelStudioProtocolTests(QueueFixture):
    """Exercise the pinned SDK over a fake HTTP transport, including PAT exchange."""
    def setUp(self):
        super().setUp()
        self.calls = []
        self.status_code = 200
        self.config = CONFIG
        self.response = {'task_ids': [42], 'task_count': 1}
        self.token_count = 0
        self.pat = jwt.encode({'exp': time.time() + 3600, 'token_type': 'refresh'}, 'test-signing-key-long-enough-for-sha256')
        self.access = jwt.encode({'exp': time.time() + 3600, 'token_type': 'access'}, 'test-signing-key-long-enough-for-sha256')
        self.http = httpx.Client(transport=httpx.MockTransport(self.handle), timeout=45)
        self.adapter = LabelStudioUploader(self.q.url, self.pat)
        self.adapter.client = LabelStudio(base_url=self.q.url + '/', api_key=self.pat,
                                           httpx_client=self.http, timeout=45, max_retries=0)
        self.q.uploader = self.adapter.upload

    def tearDown(self):
        self.http.close()
        super().tearDown()

    def handle(self, request):
        self.calls.append(request)
        if request.url.path == '/api/token/refresh/':
            self.token_count += 1
            self.assertEqual(json.loads(request.content)['refresh'], self.pat)
            return httpx.Response(200, json={'access': self.access})
        self.assertEqual(request.headers['Authorization'], 'Bearer ' + self.access)
        if self.status_code != 200:
            return httpx.Response(self.status_code, json={'detail': 'secret-test-key'})
        if request.method == 'GET':
            return httpx.Response(200, json={'id': 1, 'label_config': self.config})
        return httpx.Response(201, json=self.response)

    def test_multipart_protocol_and_refresh(self):
        self.save()
        self.q.process_one()
        self.assertEqual(self.q.status()['uploaded'], 1)
        upload = self.calls[-1]
        self.assertEqual(upload.method, 'POST')
        self.assertEqual(upload.url.path, '/api/projects/1/import')
        self.assertEqual(upload.url.params['return_task_ids'], 'true')
        self.assertIn('multipart/form-data', upload.headers['content-type'])
        self.assertIn(self.jpeg, upload.content)
        self.assertIn(f"sortie-{self.payload['session']}-{self.payload['id']}.jpg".encode(), upload.content)
        self.assertNotIn(b'annotations', upload.content)
        self.assertNotIn(b'predictions', upload.content)
        self.assertEqual(self.token_count, 1)
        # Force access-token expiry; next image must refresh the PAT again.
        self.adapter.client._client_wrapper._tokens_client._access_token = jwt.encode(
            {'exp': time.time() - 60}, 'test-signing-key-long-enough-for-sha256')
        self.payload['id'] = str(uuid.uuid4())
        self.save()
        self.q.process_one()
        self.assertEqual(self.token_count, 2)
        self.assertEqual(self.q.status()['uploaded'], 2)

    def test_bad_project_never_imports(self):
        for config in ('<View/>', CONFIG.replace('glass', 'metal'),
                       CONFIG.replace('$image', '$photo'), '<broken'):
            self.config = config
            self.save()
            self.q.retry()
            self.q.process_one()
            self.assertEqual(self.q.status()['failed'], 1)
        self.assertFalse(any(r.url.path.endswith('/import') for r in self.calls))

    def test_missing_task_id_keeps_jpeg(self):
        self.save()
        for response in ({}, {'task_ids': []}, {'task_ids': ['42']}, {'import': 123}):
            self.response = response
            self.q.retry()
            self.q.process_one()
            self.assertEqual(self.q.status()['uploaded'], 0)
            self.assertEqual(self.q.status()['failed'], 1)
            with self.q.connect() as db:
                self.assertEqual(db.execute('SELECT jpeg FROM captures').fetchone()[0], self.jpeg)

    def test_credentials_rate_limit_and_network_errors(self):
        self.save()
        for status, failed in [(401, 1), (403, 1), (404, 1), (429, 0), (503, 0)]:
            self.status_code = status
            self.q.retry()
            self.q.process_one()
            self.assertEqual(self.q.status()['failed'], failed)
            self.assertNotIn('secret-test-key', str(self.q.status()))
            self.assertNotIn(self.pat, str(self.q.status()))
        self.q.retry()
        with patch.object(self.adapter.client.projects, 'get', side_effect=httpx.ConnectError('secret-test-key')):
            self.q.process_one()
        self.assertNotIn('secret-test-key', str(self.q.status()))


if __name__ == '__main__':
    unittest.main()
