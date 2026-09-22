"""
Download the MediaPipe Hand Landmarker assets Sortie needs, so the kiosk works
without a network connection.

    python tools/fetch_hand_assets.py            # fetch anything missing
    python tools/fetch_hand_assets.py --force    # re-download everything

Everything lands in two git-ignored directories next to the app:

    static/vendor/tasks-vision/   the tasks-vision ES module + its wasm runtime
    static/models/               hand_landmarker.task

Without these files the site still runs — `static/js/hands.mjs` reports the
tracker as unavailable and `app.js` falls back to whole-frame motion detection.

Stdlib only, so this works before `pip install -r requirements.txt`.
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Pinned so a surprise upstream release can't change behaviour underneath the app.
TASKS_VISION = "1.0.1"
CDN = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VISION}"
MODEL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
    "/hand_landmarker/float16/1/hand_landmarker.task"
)

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "static" / "vendor" / "tasks-vision"
MODELS = ROOT / "static" / "models"

# (url, destination). The nosimd pair is the fallback the FilesetResolver picks
# on browsers without WebAssembly SIMD; it chooses at runtime, so ship both.
ASSETS = [
    (f"{CDN}/vision_bundle.mjs", VENDOR / "vision_bundle.mjs"),
    *[
        (f"{CDN}/wasm/{name}", VENDOR / "wasm" / name)
        for name in (
            "vision_wasm_internal.js",
            "vision_wasm_internal.wasm",
            "vision_wasm_nosimd_internal.js",
            "vision_wasm_nosimd_internal.wasm",
        )
    ],
    (MODEL, MODELS / "hand_landmarker.task"),
]


def fetch(url, dest, force):
    if dest.exists() and not force:
        return dest, dest.stat().st_size, "kept"
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # Write beside the target first so an interrupted run leaves no half file.
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as res, tmp.open("wb") as handle:
        while chunk := res.read(1 << 16):
            digest.update(chunk)
            handle.write(chunk)
    tmp.replace(dest)
    return dest, dest.stat().st_size, digest.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    total = 0
    for url, dest in ASSETS:
        try:
            path, size, note = fetch(url, dest, args.force)
        except (urllib.error.URLError, OSError) as exc:
            print(f"FAILED  {url}\n        {exc}", file=sys.stderr)
            return 1
        total += size
        print(f"{size / 1e6:8.2f} MB  {path.relative_to(ROOT)}  ({note})")

    print(f"{total / 1e6:8.2f} MB  total")
    print("\nHand detection is ready. Start the app with: python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
