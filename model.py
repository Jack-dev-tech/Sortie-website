"""
Sortie — item classifier.

This module has ONE job: turn an image into a prediction.

    classify(image) -> {"category": <one of CATEGORIES>, "confidence": 0.0..1.0, ...}

It runs any .tflite image model — a plain classifier (one score per label) or
an object detector (boxes + labels + scores, e.g. EfficientDet from TFLite
Model Maker). It reads the input size, dtype and the label list from the model
itself, so swapping models is usually just:

    1. Drop the new .tflite next to this file (or point MODEL_PATH / $SORTIE_MODEL at it).
    2. Run  `python model.py`  to see its labels.
    3. Update LABEL_MAP so every label lands in one of the four bins.

Test a model from the terminal without the webcam:

    python model.py                       # print model info + label mapping check
    python model.py photo.jpg [more.jpg]  # run predictions on image files
    python model.py --model other.tflite photo.jpg
    python model.py --threshold 0 photo.jpg   # detectors: show even the weakest boxes
"""

import io
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings — this is the only block you normally touch.
# ---------------------------------------------------------------------------

# The four bins the UI knows about (must match app.js / index.html).
CATEGORIES = ["glass", "paper", "plastic", "waste"]

# Set True to get fake predictions without a model (handy for UI work).
# Or run:  SORTIE_MOCK=1 python app.py
MOCK_MODE = os.environ.get("SORTIE_MOCK", "0") == "1"

# Which model to load. Override without editing:  SORTIE_MODEL=path/to/x.tflite python app.py
MODEL_PATH = os.environ.get("SORTIE_MODEL", str(Path(__file__).with_name("model.tflite")))

# model label  ->  bin.  Run `python model.py` to print the labels a model has.
# Labels missing here fall back to DEFAULT_CATEGORY (set it to None to ignore them).
LABEL_MAP = {
    # waste.tflite (EfficientDet, 5 labels)
    "plastic_waste": "plastic",
    "bio_waste": "waste",
    "dustbin_waste": "waste",
    "open_litter": "waste",
    "hospital_waste": "waste",
    # garbageclassification.pth (RetinaNet, 5 labels) + common TrashNet-style labels.
    "white-glass": "glass",
    "green-glass": "glass",
    "brown-glass": "glass",
    "glass": "glass",
    "paper": "paper",
    "cardboard": "paper",
    "plastic": "plastic",
    "metal": "waste",
    "trash": "waste",
}
DEFAULT_CATEGORY = "waste"

# Labels: leave None to auto-detect (a labels/labelmap .txt next to the model,
# else the list embedded in the .tflite metadata). Or hard-code the list here,
# in the model's output order, e.g. ["glass", "paper", "plastic", "waste"].
LABELS = None

# Detection models only: ignore boxes scoring below this (0..1).
SCORE_THRESHOLD = 0.20

# Float-input models only: pixel range the model expects. Most Keras/MobileNet
# exports want (-1, 1); Model Maker / TF Hub exports usually want (0, 1).
FLOAT_INPUT_RANGE = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API — app.py calls this. Extra keys in the result are fine; the UI
# only reads "category" and "confidence".
# ---------------------------------------------------------------------------
def classify(image):
    """
    image: PIL.Image (RGB).
    returns: {"category": str|None, "confidence": float, "label": str|None, "top": [...]}
    category is None when a detector sees nothing above SCORE_THRESHOLD.
    """
    if MOCK_MODE:
        return _mock_classify()
    return _get_model().predict(image)


def _mock_classify():
    import random
    bucket = int(time.time() // 5)               # rotate every ~5 s so the UI locks a result
    rng = random.Random(bucket)
    category = rng.choice(CATEGORIES)
    confidence = round(rng.uniform(0.82, 0.98), 3)
    return {"category": category, "confidence": confidence, "label": "mock", "top": []}


# ---------------------------------------------------------------------------
# TFLite wrapper
# ---------------------------------------------------------------------------
def _import_interpreter():
    """Return a TFLite Interpreter class from whichever runtime is installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter      # pip install ai-edge-litert
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter      # pip install tflite-runtime
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter                 # pip install tensorflow
        return Interpreter
    except ImportError:
        raise ImportError(
            "No TFLite runtime found. Run:  pip install ai-edge-litert   "
            "(or tflite-runtime / tensorflow)."
        )


def read_labels(model_path):
    """Find the label list for a model: sidecar .txt first, then embedded metadata."""
    if LABELS:
        return list(LABELS)
    p = Path(model_path)
    for cand in (p.with_suffix(".labels.txt"), p.with_name("labels.txt"), p.with_name("labelmap.txt")):
        if cand.exists():
            return _parse_labels(cand.read_text())
    # TFLite metadata stores associated files as a zip appended to the flatbuffer.
    try:
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.lower().endswith(".txt"):
                    return _parse_labels(z.read(name).decode("utf-8"))
    except (zipfile.BadZipFile, OSError):
        pass
    return []


def _parse_labels(text):
    labels = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept "0 glass" / "0:glass" style lines too.
        parts = line.replace(":", " ").split(None, 1)
        labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


class TFLiteModel:
    """Loads a .tflite once and turns PIL images into bin predictions."""

    def __init__(self, model_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}  — put your .tflite there, set MODEL_PATH in "
                f"model.py, or set SORTIE_MODEL=... (or SORTIE_MOCK=1 for fake predictions)."
            )
        self.path = str(model_path)
        self._lock = threading.Lock()   # interpreters are not thread-safe

        t0 = time.perf_counter()
        Interpreter = _import_interpreter()
        self.interp = Interpreter(model_path=self.path, num_threads=os.cpu_count() or 1)
        self.interp.allocate_tensors()
        self.load_seconds = time.perf_counter() - t0

        self.inp = self.interp.get_input_details()[0]
        self.outputs = self.interp.get_output_details()
        shape = list(self.inp["shape"])
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"Expected an image input of shape [1, H, W, 3], got {shape}")
        self.height, self.width = int(shape[1]), int(shape[2])
        self.labels = read_labels(self.path)
        self.kind = self._detect_kind()
        if not self.labels:
            n = self._num_classes()
            self.labels = [f"class_{i}" for i in range(n)]

    # -- model shape sniffing -------------------------------------------------
    def _detect_kind(self):
        shapes = [tuple(int(x) for x in o["shape"]) for o in self.outputs]
        if len(self.outputs) == 1 and len(shapes[0]) == 2:
            return "classifier"
        if len(self.outputs) == 4 and any(len(s) == 3 and s[-1] == 4 for s in shapes):
            return "detector"                 # TFLite detection postprocess: boxes, classes, scores, count
        if len(self.outputs) == 2 and all(len(s) == 3 for s in shapes) and any(s[-1] == 4 for s in shapes):
            return "detector-dense"           # raw boxes [1,N,4] + per-class scores [1,N,C] (no NMS op)
        raise ValueError(
            f"Unrecognised model outputs {shapes}. Expected one [1, N] tensor (classifier), "
            f"the 4 detection tensors (boxes, classes, scores, count), or boxes + class scores."
        )

    def _num_classes(self):
        if self.kind == "classifier":
            return int(self.outputs[0]["shape"][-1])
        if self.kind == "detector-dense":
            return max(int(o["shape"][-1]) for o in self.outputs if int(o["shape"][-1]) != 4) or 0
        return 0

    # -- inference ------------------------------------------------------------
    def _preprocess(self, image):
        img = image.convert("RGB").resize((self.width, self.height))
        arr = np.asarray(img)
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            lo, hi = FLOAT_INPUT_RANGE
            arr = arr.astype(np.float32) / 255.0 * (hi - lo) + lo
        elif dtype == np.int8:
            arr = (arr.astype(np.int16) - 128).astype(np.int8)
        return np.expand_dims(arr.astype(dtype), 0)

    def _run(self, image):
        x = self._preprocess(image)
        with self._lock:
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            return [np.array(self.interp.get_tensor(o["index"])) for o in self.outputs]

    def raw(self, image):
        """Classifier: one entry per label with its probability. Detector: every box, best first."""
        outs = self._run(image)
        if self.kind == "classifier":
            return self._raw_classifier(outs[0][0])
        if self.kind == "detector-dense":
            return self._raw_dense(outs)
        return self._raw_detector(outs)

    def _raw_classifier(self, scores):
        det = self.outputs[0]
        scale, zero = det["quantization"]
        if det["dtype"] != np.float32 and scale:
            scores = (scores.astype(np.float32) - zero) * scale
        scores = scores.astype(np.float32)
        if scores.min() < 0 or scores.max() > 1.0001 or abs(scores.sum() - 1) > 0.05:
            e = np.exp(scores - scores.max())         # logits -> softmax
            scores = e / e.sum()
        names = self.labels if len(self.labels) == len(scores) else [f"class_{i}" for i in range(len(scores))]
        return [{"label": n, "score": float(s), "box": None} for n, s in zip(names, scores)]

    def _raw_detector(self, outs):
        # Identify tensors by shape rather than position: exports differ in order.
        boxes = classes = scores = count = None
        flat = []
        for o, t in zip(self.outputs, outs):
            s = t.shape
            if len(s) == 3 and s[-1] == 4:
                boxes = t[0]
            elif len(s) == 1:
                count = int(t[0])
            elif len(s) == 2:
                flat.append(t[0])
        # Of the two [1, N] tensors, class ids are integer-valued.
        if len(flat) == 2:
            a, b = flat
            if np.all(a == np.round(a)) and not np.all(b == np.round(b)):
                classes, scores = a, b
            elif np.all(b == np.round(b)) and not np.all(a == np.round(a)):
                classes, scores = b, a
            else:                                     # ambiguous: TFLite default order is classes, scores
                classes, scores = a, b
        if boxes is None or classes is None:
            raise ValueError("Could not identify detector output tensors.")
        n = count if count is not None else len(scores)
        dets = []
        for i in range(min(n, len(scores))):
            score = float(scores[i])
            idx = int(classes[i])
            label = self.labels[idx] if 0 <= idx < len(self.labels) else f"class_{idx}"
            dets.append({"label": label, "score": score, "box": [float(v) for v in boxes[i]]})
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def _raw_dense(self, outs):
        """Every anchor has a score per class: report the best-scoring box for each label."""
        boxes = scores = None
        for t in outs:
            if t.shape[-1] == 4 and boxes is None:
                boxes = t[0]
            else:
                scores = t[0]
        if scores.shape[-1] == 4 and boxes.shape[-1] != 4:       # C happens to be 4: swap if needed
            boxes, scores = scores, boxes
        n_cls = scores.shape[-1]
        names = self.labels if len(self.labels) == n_cls else [f"class_{i}" for i in range(n_cls)]
        best = scores.argmax(axis=0)                                # best anchor per class
        dets = [{"label": names[c], "score": float(scores[best[c], c]),
                 "box": [float(v) for v in boxes[best[c]]]} for c in range(n_cls)]
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def predict(self, image):
        """Collapse raw label scores into one of CATEGORIES."""
        raw = self.raw(image)
        per_cat = {}
        for d in raw:
            if self.kind != "classifier" and d["score"] < SCORE_THRESHOLD:
                continue
            cat = LABEL_MAP.get(d["label"], DEFAULT_CATEGORY)
            if cat is None:
                continue
            if self.kind == "classifier":
                per_cat[cat] = per_cat.get(cat, 0.0) + d["score"]   # P(bin) = sum of its labels
            else:
                per_cat[cat] = max(per_cat.get(cat, 0.0), d["score"])
        top = [{"label": d["label"], "score": round(d["score"], 3)} for d in
               sorted(raw, key=lambda d: d["score"], reverse=True)[:3]]
        if not per_cat:
            return {"category": None, "confidence": 0.0, "label": None, "top": top}
        best = max(per_cat, key=per_cat.get)
        return {
            "category": best,
            "confidence": round(float(min(per_cat[best], 1.0)), 3),
            "label": top[0]["label"] if top else None,
            "top": top,
        }

    # -- introspection ----------------------------------------------------------
    def describe(self):
        lines = [
            f"Model:   {self.path}",
            f"Kind:    {self.kind}   (loaded in {self.load_seconds:.2f}s)",
            f"Input:   {[int(x) for x in self.inp['shape']]}  {self.inp['dtype'].__name__}",
        ]
        for o in self.outputs:
            lines.append(f"Output:  {o['name']}  {[int(x) for x in o['shape']]}  {o['dtype'].__name__}")
        lines.append(f"Labels ({len(self.labels)}):")
        problems = []
        for i, lab in enumerate(self.labels):
            cat = LABEL_MAP.get(lab)
            note = f"-> {cat}" if cat else f"-> (not in LABEL_MAP, falls back to {DEFAULT_CATEGORY})"
            if cat and cat not in CATEGORIES:
                note += "   !! not one of CATEGORIES"
                problems.append(lab)
            if not cat:
                problems.append(lab)
            lines.append(f"  {i}: {lab:<24} {note}")
        if self.kind != "classifier":
            lines.append(f"Score threshold: {SCORE_THRESHOLD}")
        lines.append("Mapping check: " + ("OK" if not problems else f"review {problems}"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = TFLiteModel(MODEL_PATH)
        return _model


# ---------------------------------------------------------------------------
# CLI:  python model.py [--model x.tflite] [image ...]
# ---------------------------------------------------------------------------
def _main(argv):
    from PIL import Image

    path = MODEL_PATH
    images = []
    it = iter(argv)
    for a in it:
        if a in ("--model", "-m"):
            path = next(it)
        elif a in ("--threshold", "-t"):
            global SCORE_THRESHOLD
            SCORE_THRESHOLD = float(next(it))
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            images.append(a)

    m = TFLiteModel(path)
    print(m.describe())
    if not images:
        return 0
    print()
    for img_path in images:
        img = Image.open(img_path)
        t0 = time.perf_counter()
        result = m.predict(img)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{img_path}:  {result['category']}  ({result['confidence']:.0%})  "
              f"label={result['label']}  {ms:.0f} ms")
        for d in result["top"]:
            print(f"    {d['label']:<24} {d['score']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
