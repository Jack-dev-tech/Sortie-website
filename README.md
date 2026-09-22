# Sortie — AI Recycling Assistant

A local web app that watches webcam and tells you which bin an item belongs
in — **glass, paper, plastic, or waste** Made for kiosk use

The UI is fully built and works right now in **mock mode** (fake predictions),
so you can see everything before your model is ready. When your `.tflite` model
is done, you drop it into one file and flip one switch.

## Run it locally

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** and allow camera access when prompted.

> Browsers only allow the camera on `localhost` or over HTTPS. `localhost` is
> fine — you don't need a certificate for local use.

## Camera privacy

People are automatically blurred in the preview **and** in every image the UI
submits to `/predict`, except for detected hands and their immediate surroundings.
Each hand keeps a clear bounding box padded by 30% on each side so nearby held
items remain visible. Faces, hair, clothing, and the rest of the body stay blurred;
detected face regions take priority even when a hand overlaps them. If a hand is
not detected, the body blur remains in effect. Detection runs in
a browser worker using locally served MediaPipe models. Camera images are never
sent to an external detection service. The existing Google Fonts links are
unrelated to camera processing.

The raw video stays hidden. Each frame is captured at up to 640 pixels wide,
detected, and blurred before it reaches the visible canvas. Motion detection
and the 320-pixel classifier JPEG both use that processed canvas. Only one frame
is processed at a time, so slower devices show fewer frames rather than applying
an old mask to a new image. Privacy processing continues during sorting results.
Loading, errors, and unsupported browsers leave the preview blank and prevent
submissions. Switching tabs stops the camera; returning starts a fresh session.

Use a current browser with Web Workers, WebAssembly, OffscreenCanvas, and
createImageBitmap support. The model and runtime assets are included under
`static/models/privacy` and `static/vendor/mediapipe`; no extra Python packages,
API keys, or runtime downloads from a CDN are required. Their versions, source
URLs, and SHA-256 hashes are recorded in `static/models/privacy/manifest.json`.

Automatic detection is best effort, **not a guarantee of anonymization**.
Small, distant, occluded, or isolated body parts can be missed. Items held close
to a person may still be blurred outside the padded hand areas, reducing
classification accuracy. The hand exceptions can also reveal nearby body pixels.
Validate with
the actual kiosk camera, distance, lighting, and users before relying on this
for privacy. The `/predict` API itself remains an image classifier; external
clients calling it directly are responsible for processing their own images.

### Privacy checks

Run the focused mask, synchronization, and failure-handling tests with Node 20+:

```bash
node --test tests/privacy.test.mjs
```

Optional Chrome integration checks use public MediaPipe sample photos and a
simulated camera, without accessing your webcam. With Google Chrome installed,
start a separate mock server:

```bash
SORTIE_MOCK=1 venv/bin/python -m flask --app app run --port 5055
```

Then, in another terminal:

```bash
npm install --prefix /tmp/sortie-browser-check playwright@1.63.0
curl -fL https://storage.googleapis.com/mediapipe-assets/pose.jpg -o /tmp/sortie-privacy-pose.jpg
curl -fL https://storage.googleapis.com/mediapipe-assets/right_hands.jpg -o /tmp/sortie-privacy-hands.jpg
PLAYWRIGHT_MODULE=/tmp/sortie-browser-check/node_modules/playwright/index.mjs node tests/privacy-browser.mjs
```

The browser checks verify real model inference, clear hands, person/background separation,
submitted JPEGs matching the preview, continuous blurring during results,
failure handling, camera denial, retry, and tab pause/resume. They save the
processed sample to `/tmp/sortie-privacy-person.png`. `SORTIE_TEST_URL` and
`SORTIE_TEST_PHOTO` override the server URL and pose sample path; `SORTIE_TEST_HANDS`
overrides the hand sample path (pixel comparisons assume these same sample photos).

## Plug in your model

Everything model-related lives in **`model.py`**. It works with any `.tflite`
image model — a classifier (one score per label) or an object detector
(EfficientDet-style boxes + labels + scores, or raw RetinaNet-style boxes +
per-class scores). Input size, dtype and the label list are read from the model
file itself.

**Swapping in a new model — the 3-step loop:**

1. Drop the file in and inspect it:

   ```bash
   cp ~/Downloads/new_model.tflite model.tflite
   python model.py                 # prints kind, input shape, labels, and a mapping check
   ```

   To try a model without renaming it: `python model.py --model ~/Downloads/x.tflite`
   or run the server with `SORTIE_MODEL=~/Downloads/x.tflite python app.py`.

2. Make sure every label the model prints lands in a bin. Edit `LABEL_MAP` in
   `model.py` (model label → `glass` / `paper` / `plastic` / `waste`). Unmapped
   labels fall back to `DEFAULT_CATEGORY`; `python model.py` warns you about them.

3. Test on a few photos, then run the site:

   ```bash
   python model.py photo1.jpg photo2.jpg   # prediction + top-3 raw labels + latency
   python app.py
   ```

**Other knobs in `model.py`:** `SCORE_THRESHOLD` (detectors: ignore weak
boxes), `FLOAT_INPUT_RANGE` (float-input models: `(0, 1)` or `(-1, 1)`),
`LABELS` (hard-code a label list if the model has none embedded), and
`SORTIE_MOCK=1` to run with fake predictions while doing UI work.

**Converting a PyTorch checkpoint (mmdet / IceVision RetinaNet):** the current
`model.tflite` was produced from `garbageclassification.pth` with
`tools/export_retinanet_tflite.py`. It needs a separate one-off Python 3.11 env
(TFLite conversion tooling is heavy and does not belong in the site's venv):

```bash
python3.11 -m venv ~/tflite-export && source ~/tflite-export/bin/activate
pip install torch torchvision litert-torch ai-edge-litert pillow
python tools/export_retinanet_tflite.py ~/Downloads/garbageclassification.pth model.tflite
deactivate
```

The script rebuilds the RetinaNet in plain PyTorch, loads the weights 1:1,
exports a uint8 224×224 model, embeds the class names, and checks the TFLite
output against PyTorch. Other architectures need a different script; the
interesting part to copy is `embed_labels()` and the output layout (boxes
`[1, N, 4]` + scores `[1, N, C]`), which `model.py` already understands.

**Labels are found in this order:** `LABELS` in `model.py` → `model.labels.txt`
/ `labels.txt` / `labelmap.txt` next to the model → the list embedded in the
`.tflite` metadata → `class_0…N`.

**The contract** — `classify()` receives a `PIL.Image` (RGB) and returns:

```python
{"category": "plastic", "confidence": 0.93, "label": "plastic_waste", "top": [...]}
```

`category` ∈ `CATEGORIES` (or `None` when a detector sees nothing) and the UI
only reads `category` + `confidence`. Note the UI ignores anything under
`CONFIDENCE_MIN` (0.75 in `static/js/app.js`); detectors usually score lower
than classifiers, so lower that while testing a detection model.

## How it fits together

```
Browser (webcam → privacy blur) ──POST /predict {image}──▶ Flask ──▶ model.classify()
      ▲                                                                    │
      └──────────  {category, confidence}  ◀───────────────────────────────┘
      │
   state machine: idle → scanning → result  (animates the matching bin)
```

## Tuning the feel

- **Frontend** (`static/js/app.js`, top of file): `MOTION_INTERVAL` (motion check
  frequency), `PREDICT_MIN_GAP` (minimum time between submissions),
  `CONFIDENCE_MIN` (ignore uncertain guesses), `STABLE_FRAMES`
  (how many matching frames lock a result), `RESULT_HOLD` (how long the verdict
  shows).
- **Colors / type** (`static/css/styles.css`, `:root`): the four category colors
  and fonts.

## Files

| File | What it is |
|------|------------|
| `app.py` | Flask server: serves the page + `/predict` |
| `model.py` | **Your integration point** — inference (mock by default) |
| `templates/index.html` | Page markup + bin icons |
| `static/css/styles.css` | Theme, layout, animations |
| `static/js/app.js` | Webcam, capture loop, state machine, animations |

## Training mode: collect and label online

Use the bottom-left **Normal / Training** switch to collect images for manual
bounding-box labeling. Training mode does not classify, recommend bins, auto-label,
or train/deploy a model. Every reload starts in Normal.

1. In [Roboflow](https://app.roboflow.com), create an **Object Detection** project.
   Use the four class names **glass**, **paper**, **plastic**, **waste** when labeling.
2. Get a private API key with access to that project. Set these variables in the
   server environment (the app does not automatically load `.env` files):

   ```sh
   export ROBOFLOW_API_KEY='your-private-key'
   export ROBOFLOW_WORKSPACE='your-workspace-slug'
   export ROBOFLOW_PROJECT='your-project-slug'
   python app.py
   ```

3. Select Training. Once camera privacy processing is ready, movement triggers at
   most one capture every three seconds. Other movement can trigger captures;
   stationary items are not repeatedly captured. **Pause capture** stops new captures.
   Leaving the tab or losing camera/privacy processing also pauses capture.
4. Choose **Open Roboflow**, find the `sortie-<session UUID>` upload batch, and open
   its images in Annotate. Draw bounding boxes and assign the four classes manually.
   Check the first uploaded image is unannotated and can be labeled before collecting
   a larger session. Roboflow may recognize an existing duplicate image; existing
   annotations on that image are not removed.

Captures contain the complete processed frame at up to 640 pixels wide, encoded as
95%-quality JPEG. They preserve the existing people blur and hand handling. Thumbnails
show the six most recently saved images from this page visit. Counts include all
captures recorded in this local queue, across sessions; pending includes failed uploads.

The server stores JPEG bytes and queue records atomically in
`data/training/captures.sqlite3` (ignored by Git). Set `SORTIE_TRAINING_DIR` to choose
another persistent local directory. Do not place it under `static/`. Successful uploads
record the Roboflow image ID and clear the JPEG blob; SQLite reuses freed storage.
Saved captures continue uploading in Normal and resume after server restarts.
An interrupted upload is eligible for recovery after its three-minute worker lease.
Network errors, HTTP 429 and server errors retry with exponential backoff, up to five
minutes. Authentication/project errors require fixing server settings, restarting,
and choosing **Retry uploads**. Pending records retain their original workspace/project. **Retry uploads** applies
the current server workspace/project to failed records, allowing you to correct a typo.

At 500 pending images, new capture pauses until uploads make room. A browser request
whose acknowledgement was lost retries the same capture UUID. Keep the page open until
it is saved; an image still waiting for the local server lives only in browser memory.
Local deduplication is by capture UUID; a remote upload interrupted after Roboflow accepted
it may be retried, relying on Roboflow's duplicate-image handling.

Server endpoints: `POST /training/captures` accepts JSON `{id, session, image}` with UUIDs
and a JPEG data URL; `GET /training/status` returns setup, counts and safe errors;
`POST /training/retry` retries pending/failed records. Credentials stay on the server.
The integration follows Roboflow's [official upload implementation](https://github.com/roboflow/roboflow-python/blob/main/roboflow/adapters/rfapi.py):
image-only multipart upload to `/dataset/{project}/upload`, with a session batch.

Run the queue/API checks without Roboflow credentials:

```sh
python -m unittest discover -s tests -p 'test_training.py'
node --test tests/privacy.test.mjs
PLAYWRIGHT_MODULE=/tmp/sortie-browser-check/node_modules/playwright/index.mjs node tests/training-browser.mjs
```

The browser checks require a local server at `http://127.0.0.1:5055` (or
`SORTIE_TEST_URL`), Playwright and Chrome. They mock upload responses and use controlled
processed frames; the existing `tests/privacy-browser.mjs` separately checks real privacy
models. Live upload verification requires your configured Roboflow project and key.
