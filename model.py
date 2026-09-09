"""
Sortie — item classifier.

This module has ONE job: turn an image into a prediction.

    classify(image) -> {"category": <one of CATEGORIES>, "confidence": 0.0..1.0}

Right now it runs in MOCK_MODE and returns fake-but-plausible predictions so the
whole website works before your model exists. When your .tflite model is ready:

    1. Put your model file next to this script (e.g. "model.tflite").
    2. Fill in _load_model() and _run_model() below (a working reference is
       included, commented out).
    3. Set MOCK_MODE = False.

That's the only switch you need to flip.
"""

import random
import time

# The four bins. Order matters if your model outputs a class index:
# index 0 -> "glass", 1 -> "paper", 2 -> "plastic", 3 -> "waste".
# Reorder this list to match YOUR model's output order.
CATEGORIES = ["glass", "paper", "plastic", "waste"]

# Flip to False once _load_model()/_run_model() are wired to your .tflite file.
MOCK_MODE = True

MODEL_PATH = "model.tflite"


# ---------------------------------------------------------------------------
# Public API — this is what app.py calls. You shouldn't need to change it.
# ---------------------------------------------------------------------------
def classify(image):
    """
    image: a PIL.Image in RGB mode (already decoded from the webcam frame).
    returns: {"category": str, "confidence": float}  where category is in CATEGORIES.
    """
    if MOCK_MODE:
        return _mock_classify()
    return _run_model(image)


# ---------------------------------------------------------------------------
# MOCK MODE — delete this once your model is in. It holds one category steady
# for a few seconds at a time so the UI's "stable result" animation triggers
# naturally, then rotates to another category.
# ---------------------------------------------------------------------------
def _mock_classify():
    bucket = int(time.time() // 5)          # changes every ~5 seconds
    rng = random.Random(bucket)             # same answer within a bucket
    category = rng.choice(CATEGORIES)
    confidence = round(rng.uniform(0.82, 0.98), 3)
    return {"category": category, "confidence": confidence}


# ---------------------------------------------------------------------------
# REAL MODEL — fill these in. Reference implementation is commented below.
# ---------------------------------------------------------------------------
_interpreter = None  # loaded once, reused for every frame


def _load_model():
    """Load the .tflite interpreter once and cache it in _interpreter."""
    global _interpreter
    if _interpreter is not None:
        return _interpreter

    # --- REFERENCE (uncomment + adjust): ---
    # try:
    #     from tflite_runtime.interpreter import Interpreter
    # except ImportError:
    #     from tensorflow.lite import Interpreter
    # _interpreter = Interpreter(model_path=MODEL_PATH)
    # _interpreter.allocate_tensors()
    # return _interpreter

    raise NotImplementedError(
        "Wire up _load_model() to your .tflite file, then set MOCK_MODE = False."
    )


def _run_model(image):
    """Preprocess `image`, run inference, return {category, confidence}."""
    interpreter = _load_model()

    # --- REFERENCE (uncomment + adjust to YOUR model's input size / dtype) ---
    # import numpy as np
    #
    # inp = interpreter.get_input_details()[0]
    # out = interpreter.get_output_details()[0]
    #
    # _, height, width, _ = inp["shape"]              # e.g. (1, 224, 224, 3)
    # img = image.convert("RGB").resize((width, height))
    # arr = np.asarray(img)
    #
    # if inp["dtype"] == np.float32:                  # float model: normalize 0..1
    #     arr = (arr.astype(np.float32) / 255.0)
    # arr = np.expand_dims(arr, axis=0).astype(inp["dtype"])
    #
    # interpreter.set_tensor(inp["index"], arr)
    # interpreter.invoke()
    # scores = interpreter.get_tensor(out["index"])[0]  # shape (num_classes,)
    #
    # idx = int(np.argmax(scores))
    # return {"category": CATEGORIES[idx], "confidence": float(scores[idx])}

    raise NotImplementedError(
        "Wire up _run_model() to your model's preprocessing + output mapping."
    )
