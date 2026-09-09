# Sortie — AI Recycling Assistant

A local web app that watches your webcam and tells you which bin an item belongs
in — **glass, paper, plastic, or waste** — with a friendly, animated kiosk UI
inspired by the Oscar Sort bin.

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

## Plug in your model

Everything model-related lives in **`model.py`**. You only touch that one file.

1. Put your model file in the project folder (default name: `model.tflite`).
2. Open `model.py` and:
   - Make sure `CATEGORIES` is in the **same order** your model outputs
     (index 0 → first item, etc.).
   - Fill in `_load_model()` and `_run_model()` — a working reference is right
     there, commented out. Match your model's input size and preprocessing.
   - Set `MOCK_MODE = False`.
3. Install a runtime — uncomment one line in `requirements.txt`:
   - `tflite-runtime` (small, just runs `.tflite` models), or
   - `tensorflow` (full package, also works).

That's it. No frontend changes needed.

**The contract** — `classify()` receives a `PIL.Image` (RGB) and must return:

```python
{"category": "plastic", "confidence": 0.93}   # category ∈ CATEGORIES, confidence 0..1
```

## How it fits together

```
Browser (webcam)  ──POST /predict {image}──▶  Flask (app.py)  ──▶  model.classify()
      ▲                                                                    │
      └──────────  {category, confidence}  ◀───────────────────────────────┘
      │
   state machine: idle → scanning → result  (animates the matching bin)
```

## Tuning the feel

- **Frontend** (`static/js/app.js`, top of file): `CAPTURE_INTERVAL` (how often
  frames are sent), `CONFIDENCE_MIN` (ignore uncertain guesses), `STABLE_FRAMES`
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
