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
import requests
from training import CaptureQueue, UploadError, install_training


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.q = CaptureQueue(self.tmp.name, 'secret-test-key', 'workspace', 'project', uploader=lambda row: 'rf-image-id')
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

    def test_success_dedupe_and_jpeg_removal(self):
        self.assertEqual(self.save().status_code, 201)
        self.assertEqual(self.save().status_code, 200)
        self.assertTrue(self.q.process_one())
        self.assertEqual(self.save().json['state'], 'uploaded')
        with self.q.connect() as db:
            row = db.execute('SELECT * FROM captures').fetchone()
            self.assertIsNone(row['jpeg'])
            self.assertEqual(row['roboflow_id'], 'rf-image-id')
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
        recovered = CaptureQueue(self.tmp.name, 'key', 'workspace', 'project', uploader=lambda row: 'recovered')
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
        self.q.project = 'corrected-project'
        self.assertEqual(self.client.post('/training/retry').status_code, 200)
        with self.q.connect() as db:
            self.assertEqual(db.execute('SELECT project FROM captures').fetchone()[0], 'corrected-project')
        self.q.uploader = lambda row: 'ok'
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

    @patch('training.requests.post')
    @patch('training.requests.get')
    def test_upload_protocol_has_no_annotations(self, get, post):
        get.return_value = Mock(status_code=200, ok=True, json=lambda: {'project': {'type': 'object-detection'}})
        post.return_value = Mock(status_code=200, ok=True, json=lambda: {'success': True, 'id': 'remote'})
        self.q.uploader = self.q.upload
        self.save(); self.q.process_one()
        kwargs = post.call_args.kwargs
        self.assertEqual(set(kwargs['data']), {'name', 'split'})
        self.assertEqual(set(kwargs['files']), {'file'})
        self.assertEqual(kwargs['files']['file'][1], self.jpeg)
        self.assertEqual(kwargs['params']['batch'], 'sortie-' + self.payload['session'])
        self.assertEqual(self.q.status()['uploaded'], 1)

    @patch('training.requests.get')
    def test_credentials_rate_limit_and_network_errors(self, get):
        self.q.uploader = self.q.upload
        self.save()
        for status, failed in [(401, 1), (403, 1), (404, 1), (429, 0), (503, 0)]:
            self.q.retry()
            get.return_value = Mock(status_code=status, ok=False)
            self.q.process_one()
            self.assertEqual(self.q.status()['failed'], failed)
            self.assertNotIn('secret-test-key', str(self.q.status()))
        self.q.retry()
        get.side_effect = requests.ConnectionError('secret-test-key')
        self.q.process_one()
        self.assertNotIn('secret-test-key', str(self.q.status()))


if __name__ == '__main__':
    unittest.main()
