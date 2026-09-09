"""
Sortie — local Flask server.

Serves the web UI and a single /predict endpoint that turns a webcam frame
into a bin recommendation. Run it and open http://localhost:5000.

    python app.py
"""

import base64
import binascii
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", categories=model.CATEGORIES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,...." }
    Returns JSON:  { "category": "plastic", "confidence": 0.93 }
    On a bad frame it returns a 400 with a message instead of crashing the loop.
    """
    data = request.get_json(silent=True) or {}
    data_url = data.get("image", "")

    if not data_url:
        return jsonify(error="No image provided."), 400

    # Strip the "data:image/...;base64," prefix if present.
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        raw = base64.b64decode(data_url, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError):
        return jsonify(error="Could not decode image."), 400

    try:
        result = model.classify(image)
    except NotImplementedError as exc:
        return jsonify(error=str(exc)), 501
    except Exception as exc:  # keep the frame loop alive on unexpected errors
        app.logger.exception("Classification failed")
        return jsonify(error=f"Classification failed: {exc}"), 500

    return jsonify(result)


if __name__ == "__main__":
    # host=127.0.0.1 keeps it local-only. Change to 0.0.0.0 to reach it from
    # another device on your network (needed for a real kiosk / tablet).
    app.run(host="127.0.0.1", port=5000, debug=True)
