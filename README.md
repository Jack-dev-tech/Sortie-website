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
python tools/fetch_hand_assets.py   # one-time, ~31 MB — see "Hand detection"
python app.py
```

Then open **http://localhost:5000** and allow camera access when prompted.

> Browsers only allow the camera on `localhost` or over HTTPS. `localhost` is
> fine — you don't need a certificate for local use.

## Hand detection

Sortie watches for a **hand**, not for movement. In Normal mode it classifies
whenever a hand is in frame — including an item held perfectly still. In Training
mode it captures when a hand is *moving* an item, so holding still doesn't fill
Label Studio with near-identical frames. The red box hugs the hand and whatever is
moving within its reach; a passer-by, a shifting shadow or your own torso never
widens it. An item shown **without** a hand (set on a table, on a conveyor)
triggers nothing.

The detector is MediaPipe's Hand Landmarker, running in the browser. Its assets
are vendored rather than loaded from a CDN so a kiosk works offline:

```bash
python tools/fetch_hand_assets.py          # fetch anything missing
python tools/fetch_hand_assets.py --force  # re-download
```

It writes `static/vendor/tasks-vision/` (the ES module + wasm runtime) and
`static/models/hand_landmarker.task`, both git-ignored. The script is stdlib-only,
so it runs before `pip install`. Pin a different `tasks-vision` release by editing
`TASKS_VISION` at the top of the script.

**If the assets are missing** — or WebGL and the CPU delegate both fail — the site
still works: `static/js/hands.mjs` reports the tracker as unavailable, the console
says so once, and `app.js` reverts to the whole-frame motion detection it used
before hand tracking existed.

## Plug in your model

Everything model-related lives in **`model.py`**. It works with any `.tflite`
image model — a classifier (one score per label) or an object detector
(EfficientDet-style boxes + labels + scores, raw RetinaNet-style boxes +
per-class scores, or an Ultralytics YOLO head: one fused `[1, 4+C, N]` tensor).
Input size, dtype, layout (NHWC or YOLO's NCHW) and the label list are all read
from the model file itself.

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
boxes, `$SORTIE_SCORE_THRESHOLD`), `CONFIDENCE_MIN` (how sure the UI has to be
before it shows a verdict, `$SORTIE_CONFIDENCE_MIN` — the server hands it to
`app.js`, so there is no JS to edit), `LETTERBOX` (`None` = auto: pad detectors
to a square instead of squashing them, which is what YOLO expects),
`FLOAT_INPUT_RANGE` (float-input models: `(0, 1)` or `(-1, 1)`), `LABELS`
(hard-code a label list if the model has none embedded), and `SORTIE_MOCK=1` to
run with fake predictions while doing UI work.

**When the UI just sits on "Scanning…":** it means nothing cleared the
confidence chain — `SCORE_THRESHOLD`, then `CONFIDENCE_MIN` twice in a row
(`STABLE_FRAMES` in `app.js`). A no-match is silent by design, so turn the
lights on:

```bash
SORTIE_DEBUG=1 python app.py
```

Every prediction then logs its raw scores *before* thresholding, and says so
when a result was dropped for being under the bar:

```
predict: plastic 0.15   Plastic=0.155 Waste=0.020 Paper=0.019  (41 ms, thresh 0.05, letterbox)
                                                   [below CONFIDENCE_MIN 0.45, UI will ignore]
```

It also keeps the last 200 frames in `data/debug/`, so you can replay exactly
what the model was shown and try a different bar against it:

```bash
python model.py data/debug/<file>.jpg
SORTIE_SCORE_THRESHOLD=0.05 python model.py data/debug/<file>.jpg
```

**Converting a PyTorch checkpoint (mmdet / IceVision RetinaNet):** an earlier
`model.tflite` was produced from `garbageclassification.pth` with
`tools/export_retinanet_tflite.py` (the current one is an Ultralytics YOLO
export). It needs a separate one-off Python 3.11 env
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
/ `labels.txt` / `labelmap.txt` next to the model → a `labels.txt` embedded in
the `.tflite` metadata → the `names` map in an embedded `metadata.json`
(Ultralytics exports) → `class_0…N`. `LABEL_MAP` lookups ignore case, so an
export that capitalises its labels (`Glass`, `Metal`, …) needs no extra entries.

**The contract** — `classify()` receives a `PIL.Image` (RGB) and returns:

```python
{"category": "plastic", "confidence": 0.93, "label": "plastic_waste", "top": [...]}
```

`category` ∈ `CATEGORIES` (or `None` when a detector sees nothing) and the UI
only reads `category` + `confidence`. Note the UI ignores anything under
`CONFIDENCE_MIN` (0.45, set in `model.py` and passed to the page) — detectors
score lower than classifiers, which is why it sits well under the 0.75 a
classifier could carry. `SORTIE_DEBUG=1` shows what the scores actually are.

## How it fits together

```
Browser (webcam) ──────────────POST /predict {image}──▶ Flask ──▶ model.classify()
      ▲                                                                    │
      └──────────  {category, confidence}  ◀───────────────────────────────┘
      │
   state machine: idle → scanning → result  (animates the matching bin)
```

## Tuning the feel

- **Frontend** (`static/js/app.js`, top of file): `TRACK_INTERVAL` (how often
  hands and motion are checked), `HAND_REACH` (how far past the hand to look for
  the item it holds — raise it for long items, lower it if the box picks up
  background), `MOTION_AREA_MIN_LOCAL` (how much of that region must change to
  call the item "moving", which is what gates training captures),
  `PREDICT_MIN_GAP` (minimum time between submissions), `FRAME_WIDTH` (width of
  the frame sent to the model), `STABLE_FRAMES` (how many matching frames lock a
  result),
  `RESULT_HOLD` (how long the verdict shows). `MOTION_PIXEL_DELTA` and
  `MOTION_AREA_MIN` tune the raw frame-diff and the no-hand fallback.
  `HAND_PAD` (`static/js/hands.mjs`) pads the box around the landmarks.
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
| `static/js/hands.mjs` | MediaPipe Hand Landmarker wrapper (hand boxes) |
| `tools/fetch_hand_assets.py` | One-time download of the hand-tracking assets |

## Training mode: collect and label online

Use the bottom-left **Normal / Training** switch to collect images for manual
bounding-box labeling. Training mode does not classify, recommend bins, auto-label,
or train/deploy a model. Every reload starts in Normal.

Use **Label Studio Community Edition** on the same Pi or mini PC as Sortie.
Connect from another computer through private Tailscale HTTPS access. Follow the
[host installation and remote annotation guide](deploy/label-studio/README.md)
for persistent storage, project setup, startup services, backup and verification.
The host must stay on; this Mac does not need to stay on.

For an already running Label Studio server, set these on the Sortie server:

```sh
export LABEL_STUDIO_URL='http://127.0.0.1:8080'
export LABEL_STUDIO_PUBLIC_URL='https://sortie.YOUR-TAILNET.ts.net'
export LABEL_STUDIO_API_KEY='your-personal-access-token'
export LABEL_STUDIO_PROJECT_ID='1'
python app.py
```

Use the supplied [bounding-box configuration](deploy/label-studio/label-config.xml)
in the Label Studio project. URLs must be HTTP(S) origins without paths, query
strings or embedded credentials. The public URL defaults to the API URL for local
development; set it explicitly for remote annotation. Variables must be exported;
the app does not automatically load `.env` files. Credentials stay on the server.

Select **Training**. A hand moving an item triggers at most one capture every
three seconds. **Pause capture**, leaving the tab, or losing the camera stops
new captures; saved images continue uploading even in Normal mode. Choose
**Open Label Studio** to draw boxes manually using **glass, paper, plastic, waste**.
Check one image loads and an annotation saves from the remote computer before
collecting a large session. No predictions or annotations are sent by Sortie.

Captures contain the complete camera frame at up to 640 pixels wide, encoded as
95%-quality JPEG. Thumbnails show the six most recently saved images from this
page visit. Uploaded filenames include both session and capture UUIDs:
`sortie-<session UUID>-<capture UUID>.jpg`.

The durable queue lives in `data/training/captures.sqlite3` (Git-ignored), or the
persistent directory specified by `SORTIE_TRAINING_DIR`. Never put it under
`static/`. Once Label Studio confirms a task ID, Sortie records it and clears its
JPEG blob. Label Studio keeps the image in its own storage, so annotation does
not depend on the capture browser remaining open. Back up Label Studio's whole
data directory, including uploaded images.

Queue records survive restarts. Interrupted workers recover after a three-minute
lease. Network failures, HTTP 408/429 and server errors retry with exponential
backoff up to five minutes. Authentication/project errors require correcting the
configuration, restarting Sortie, then choosing **Retry uploads**. Records retain
their destination; changed destinations are blocked until explicit retry retargets
the failed records. The queue stops new capture at 500 pending images or when an
upload requires attention.

Browser retries reuse the same capture UUID, preventing duplicate local records.
A connection failure after Label Studio accepted an import can cause a duplicate
remote task on retry; use the capture UUID in the filename to identify it. An
import response without one confirmed task ID retains the JPEG and requires
manual attention before retrying. This adapter targets Community Edition's
synchronous imports, not Enterprise asynchronous import jobs.

Existing Roboflow queue databases migrate automatically on startup. Unsent images
move to the configured Label Studio project once configuration is available.
Completed Roboflow records remain historical records; their images and annotations
are not imported. Counts include historical uploaded records. Stop any old uploader
before migration and keep a backup of the entire queue directory.

Server endpoints remain: `POST /training/captures` accepts JSON `{id, session, image}`
with UUIDs and a JPEG data URL; `GET /training/status` returns setup, counts, a
browser-accessible `project_url` and safe errors; `POST /training/retry` retries
pending/failed records.

Run queue/API checks without a real server or credentials:

```sh
python -m unittest discover -s tests -p 'test_training.py'
PLAYWRIGHT_MODULE=/tmp/sortie-browser-check/node_modules/playwright/index.mjs node tests/training-browser.mjs
```

Browser checks require Sortie at `http://127.0.0.1:5055` (or `SORTIE_TEST_URL`),
Playwright and Chrome. They mock uploads and drive a fake camera with a test hand
tracker. Protocol tests use the pinned official SDK with a fake HTTP transport,
including personal-token refresh. Live remote annotation still requires the
configured host and a second computer.
