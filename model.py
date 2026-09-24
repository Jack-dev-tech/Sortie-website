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
import json
import logging
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

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
    "waste": "waste",
}
DEFAULT_CATEGORY = "waste"

# Labels: leave None to auto-detect (a labels/labelmap .txt next to the model,
# else the list embedded in the .tflite metadata). Or hard-code the list here,
# in the model's output order, e.g. ["glass", "paper", "plastic", "waste"].
LABELS = None

# Detection models only: ignore boxes scoring below this (0..1).
SCORE_THRESHOLD = float(os.environ.get("SORTIE_SCORE_THRESHOLD", "0.20"))

# How sure the UI has to be before it shows a verdict (0..1). Detectors score
# lower than classifiers, so this sits well below the old classifier-era 0.75.
# Tune without editing JS:  SORTIE_CONFIDENCE_MIN=0.3 python app.py
CONFIDENCE_MIN = float(os.environ.get("SORTIE_CONFIDENCE_MIN", "0.45"))

# Log every prediction (and dump the frames it saw) — see README.
# Turn on with:  SORTIE_DEBUG=1 python app.py
DEBUG = os.environ.get("SORTIE_DEBUG", "0") == "1"

log = logging.getLogger("sortie")
if DEBUG and not log.handlers:
    # Give the logger its own stderr handler: nothing else configures logging, so
    # without this INFO records fall through to the WARNING-only last-resort
    # handler and the debug lines never appear.
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False        # don't double-print if something configures root

# Float-input models only: pixel range the model expects. Most Keras/MobileNet
# exports want (-1, 1); Model Maker / TF Hub exports usually want (0, 1).
FLOAT_INPUT_RANGE = (0.0, 1.0)

# Resize style. Detectors (YOLO especially) are trained on letterboxed frames —
# aspect preserved, the leftover padded grey — so squashing a 16:9 webcam frame
# into a square costs real confidence. None = auto: letterbox everything except
# plain classifiers. True / False force it.
LETTERBOX = None
LETTERBOX_FILL = (114, 114, 114)      # the grey Ultralytics pads with


def bin_for(label):
    """LABEL_MAP lookup that ignores case and stray whitespace.

    Exports capitalise their labels ("Glass", "Metal", ...), so a literal
    LABEL_MAP.get() would miss every one of them and send the lot to
    DEFAULT_CATEGORY. Returns None when the label is mapped to None on purpose.
    """
    lookup = {str(k).strip().lower(): v for k, v in LABEL_MAP.items()}
    key = str(label).strip().lower()
    return lookup[key] if key in lookup else DEFAULT_CATEGORY


def in_label_map(label):
    """True when LABEL_MAP names this label outright (used by describe())."""
    return str(label).strip().lower() in {str(k).strip().lower() for k in LABEL_MAP}


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
            # Ultralytics (YOLO) exports ship metadata.json with a {"0": "Glass", ...} map.
            for name in z.namelist():
                if name.lower().endswith("metadata.json"):
                    labels = _parse_ultralytics_names(z.read(name))
                    if labels:
                        return labels
    except (zipfile.BadZipFile, OSError):
        pass
    return []


def _parse_ultralytics_names(blob):
    """Pull the ordered class list out of an Ultralytics metadata.json blob."""
    try:
        names = json.loads(blob.decode("utf-8"))["names"]
        if isinstance(names, dict):                      # {"0": "Glass", "1": "Metal", ...}
            return [names[k] for k in sorted(names, key=lambda k: int(k))]
        return list(names)                               # already a list
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
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
        shape = [int(x) for x in self.inp["shape"]]
        # Keras/TFLite exports are NHWC; Ultralytics (YOLO) exports are NCHW.
        if len(shape) == 4 and shape[-1] == 3:
            self.layout = "NHWC"
            self.height, self.width = shape[1], shape[2]
        elif len(shape) == 4 and shape[1] == 3:
            self.layout = "NCHW"
            self.height, self.width = shape[2], shape[3]
        else:
            raise ValueError(
                f"Expected an image input of shape [1, H, W, 3] or [1, 3, H, W], got {shape}"
            )
        self.labels = read_labels(self.path)
        self.kind = self._detect_kind()
        if not self.labels:
            n = self._num_classes()
            self.labels = [f"class_{i}" for i in range(n)]
        self.letterbox = self.kind != "classifier" if LETTERBOX is None else bool(LETTERBOX)
        # Set per frame by _fit(); _unletterbox() undoes them for the boxes.
        self._fit_w, self._fit_h = self.width, self.height
        self._pad_x = self._pad_y = 0

    # -- model shape sniffing -------------------------------------------------
    def _detect_kind(self):
        shapes = [tuple(int(x) for x in o["shape"]) for o in self.outputs]
        if len(self.outputs) == 1 and len(shapes[0]) == 2:
            return "classifier"
        if len(self.outputs) == 1 and len(shapes[0]) == 3 and 4 not in shapes[0][1:]:
            return "yolo"                     # one fused [1, 4+C, N] (or [1, N, 4+C]) tensor
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
        if self.kind == "yolo":
            d1, d2 = (int(x) for x in self.outputs[0]["shape"][1:])
            return min(d1, d2) - 4            # the short axis is 4 box rows + one row per class
        return 0

    # -- inference ------------------------------------------------------------
    def _fit(self, image):
        """Resize to the model's input, letterboxed or squashed.

        Letterboxing keeps the aspect ratio and pads the rest — a 16:9 webcam
        frame squashed into a square is a 1.78x distortion, which costs a
        detector real confidence. Records the scale/pad so _unletterbox() can
        map boxes back onto the original frame.
        """
        img = image.convert("RGB")
        if not self.letterbox:
            self._fit_w, self._fit_h = self.width, self.height
            self._pad_x = self._pad_y = 0
            return img.resize((self.width, self.height))

        w, h = img.size
        scale = min(self.width / w, self.height / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = Image.new("RGB", (self.width, self.height), LETTERBOX_FILL)
        pad_x, pad_y = (self.width - new_w) // 2, (self.height - new_h) // 2
        canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
        self._fit_w, self._fit_h = new_w, new_h
        self._pad_x, self._pad_y = pad_x, pad_y
        return canvas

    def _unletterbox(self, x, y):
        """Padded-canvas coords (0..1) -> original-frame coords (0..1).

        A box may overhang into the padding, so clamp it back onto the frame.
        """
        if not self.letterbox:
            return x, y
        x = (x * self.width - self._pad_x) / self._fit_w
        y = (y * self.height - self._pad_y) / self._fit_h
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _preprocess(self, image):
        img = self._fit(image)
        arr = np.asarray(img)
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            lo, hi = FLOAT_INPUT_RANGE
            arr = arr.astype(np.float32) / 255.0 * (hi - lo) + lo
        elif dtype == np.int8:
            arr = (arr.astype(np.int16) - 128).astype(np.int8)
        if self.layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))
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
        if self.kind == "yolo":
            return self._raw_yolo(outs)
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
            dets.append({"label": label, "score": score, "box": self._fix_box(boxes[i])})
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
                 "box": self._fix_box(boxes[best[c]])} for c in range(n_cls)]
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def _raw_yolo(self, outs):
        """Ultralytics head: one [1, 4+C, N] tensor of cx,cy,w,h + a score per class.

        Like _raw_dense, we keep the best-scoring anchor per class. predict() only
        wants the strongest score per bin, so there is nothing for NMS to do.
        """
        t = outs[0][0]
        if t.shape[0] < t.shape[1]:
            t = t.T                                      # [4+C, N] -> [N, 4+C]
        boxes, scores = t[:, :4], t[:, 4:]               # v8 heads are already 0..1, no objectness
        n_cls = scores.shape[-1]
        names = self.labels if len(self.labels) == n_cls else [f"class_{i}" for i in range(n_cls)]
        best = scores.argmax(axis=0)                     # best anchor per class
        dets = [{"label": names[c], "score": float(scores[best[c], c]),
                 "box": self._xywh_to_box(boxes[best[c]])} for c in range(n_cls)]
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def _fix_box(self, box):
        """[ymin, xmin, ymax, xmax] off the padded canvas -> the same on the frame."""
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        xmin, ymin = self._unletterbox(xmin, ymin)
        xmax, ymax = self._unletterbox(xmax, ymax)
        return [ymin, xmin, ymax, xmax]

    def _xywh_to_box(self, xywh):
        """cx,cy,w,h -> the normalised [ymin, xmin, ymax, xmax] the other heads return."""
        cx, cy, w, h = (float(v) for v in xywh)
        if max(cx, cy, w, h) > 1.5:                      # pixel-unit export: scale to 0..1
            cx, w = cx / self.width, w / self.width
            cy, h = cy / self.height, h / self.height
        # Coords are relative to the padded canvas; put them back on the frame.
        xmin, ymin = self._unletterbox(cx - w / 2, cy - h / 2)
        xmax, ymax = self._unletterbox(cx + w / 2, cy + h / 2)
        return [ymin, xmin, ymax, xmax]

    def predict(self, image):
        """Collapse raw label scores into one of CATEGORIES."""
        t0 = time.perf_counter()
        raw = self.raw(image)
        per_cat = {}
        for d in raw:
            if self.kind != "classifier" and d["score"] < SCORE_THRESHOLD:
                continue
            cat = bin_for(d["label"])
            if cat is None:
                continue
            if self.kind == "classifier":
                per_cat[cat] = per_cat.get(cat, 0.0) + d["score"]   # P(bin) = sum of its labels
            else:
                per_cat[cat] = max(per_cat.get(cat, 0.0), d["score"])
        top = [{"label": d["label"], "score": round(d["score"], 3)} for d in
               sorted(raw, key=lambda d: d["score"], reverse=True)[:3]]
        if not per_cat:
            result = {"category": None, "confidence": 0.0, "label": None, "top": top}
        else:
            best = max(per_cat, key=per_cat.get)
            result = {
                "category": best,
                "confidence": round(float(min(per_cat[best], 1.0)), 3),
                "label": top[0]["label"] if top else None,
                "top": top,
            }
        if DEBUG:
            self._log(result, (time.perf_counter() - t0) * 1000)
        return result

    def _log(self, result, ms):
        """One line per prediction, with the raw scores *before* thresholding.

        A category of None still prints how close it got — that is the number
        you need to pick SCORE_THRESHOLD / CONFIDENCE_MIN.
        """
        scores = "  ".join(f"{d['label']}={d['score']:.3f}" for d in result["top"])
        verdict = (f"{result['category']} {result['confidence']:.2f}"
                   if result["category"] else "-- no match")
        gate = "" if not result["category"] else (
            "" if result["confidence"] >= CONFIDENCE_MIN
            else f"  [below CONFIDENCE_MIN {CONFIDENCE_MIN}, UI will ignore]")
        log.info(
            "predict: %-18s %s  (%.0f ms, thresh %.2f%s)%s",
            verdict, scores, ms, SCORE_THRESHOLD,
            ", letterbox" if self.letterbox else "", gate,
        )

    # -- introspection ----------------------------------------------------------
    def describe(self):
        lines = [
            f"Model:   {self.path}",
            f"Kind:    {self.kind}   (loaded in {self.load_seconds:.2f}s)",
            f"Input:   {[int(x) for x in self.inp['shape']]}  {self.inp['dtype'].__name__}  "
            f"({self.layout}, {'letterboxed' if self.letterbox else 'squashed'})",
        ]
        for o in self.outputs:
            lines.append(f"Output:  {o['name']}  {[int(x) for x in o['shape']]}  {o['dtype'].__name__}")
        lines.append(f"Labels ({len(self.labels)}):")
        problems = []
        for i, lab in enumerate(self.labels):
            cat = bin_for(lab) if in_label_map(lab) else None
            note = f"-> {cat}" if cat else f"-> (not in LABEL_MAP, falls back to {DEFAULT_CATEGORY})"
            if cat and cat not in CATEGORIES:
                note += "   !! not one of CATEGORIES"
                problems.append(lab)
            if not cat:
                problems.append(lab)
            lines.append(f"  {i}: {lab:<24} {note}")
        if self.kind != "classifier":
            lines.append(f"Score threshold: {SCORE_THRESHOLD}   UI confidence min: {CONFIDENCE_MIN}")
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
