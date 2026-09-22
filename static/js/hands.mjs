// Finds hands in the live video with MediaPipe's Hand Landmarker, so the rest
// of the app can react to a held item instead of to any movement at all.
//
// The assets are vendored (tools/fetch_hand_assets.py) rather than loaded from a
// CDN, so a kiosk works offline. When they are missing the tracker stays
// unavailable and app.js falls back to whole-frame motion detection.

const WASM_DIR = "/static/vendor/tasks-vision/wasm";
const BUNDLE = "/static/vendor/tasks-vision/vision_bundle.mjs";
const MODEL = "/static/models/hand_landmarker.task";

const MAX_HANDS = 2;
const HAND_PAD = 0.25; // grow the landmark hull by this fraction of its larger side

export class HandTracker {
  constructor() {
    this.available = false;
    this.landmarker = null;
    this.loading = null;
    this.hands = [];
    this.lastVideoTime = -1;
    this.lastTimestamp = 0;
  }

  // Idempotent. Resolves once the model is usable or once we know it never will be.
  load() {
    this.loading ||= this._load().catch((err) => {
      console.warn(
        "Sortie: hand tracking unavailable, falling back to whole-frame motion. " +
          "Run `python tools/fetch_hand_assets.py` to install it.",
        err
      );
    });
    return this.loading;
  }

  async _load() {
    if (testHands()) {
      // A test harness is driving hand positions; skip the 20MB download.
      this.available = true;
      return;
    }
    const { FilesetResolver, HandLandmarker } = await import(BUNDLE);
    const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
    const options = {
      baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: MAX_HANDS,
    };
    try {
      this.landmarker = await HandLandmarker.createFromOptions(fileset, options);
    } catch (err) {
      // No WebGL (remote desktop, some kiosks, headless browsers) — CPU still works.
      console.warn("Sortie: hand tracking falling back to the CPU delegate.", err);
      options.baseOptions.delegate = "CPU";
      this.landmarker = await HandLandmarker.createFromOptions(fileset, options);
    }
    this.available = true;
  }

  // → [{ x, y, w, h }] normalised to 0..1 in raw (unmirrored) camera coordinates,
  // the same space detectMotion() and drawMotionBoxes() work in.
  detect(video) {
    const fake = testHands();
    if (fake) return (this.hands = typeof fake === "function" ? fake() : fake);
    if (!this.landmarker || !video.videoWidth) return (this.hands = []);

    // detectForVideo wants a fresh frame and a strictly increasing timestamp;
    // re-running it on the same frame throws, so reuse the last answer instead.
    if (video.currentTime === this.lastVideoTime) return this.hands;
    this.lastVideoTime = video.currentTime;
    this.lastTimestamp = Math.max(performance.now(), this.lastTimestamp + 1);

    let result;
    try {
      result = this.landmarker.detectForVideo(video, this.lastTimestamp);
    } catch (err) {
      console.warn("Sortie: hand detection failed on this frame.", err);
      return this.hands;
    }
    return (this.hands = (result?.landmarks ?? []).map(boxOf).filter(Boolean));
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
    this.available = false;
    this.hands = [];
  }
}

// The bounding box of one hand's 21 landmarks, padded so it covers the whole
// hand rather than just the joint positions.
function boxOf(landmarks) {
  if (!landmarks?.length) return null;
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const { x, y } of landmarks) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = Math.max(maxX - minX, maxY - minY) * HAND_PAD;
  const x = Math.max(0, minX - pad);
  const y = Math.max(0, minY - pad);
  return {
    x,
    y,
    w: Math.min(1, maxX + pad) - x,
    h: Math.min(1, maxY + pad) - y,
  };
}

// Test seam, in the style of window.testStill / window.fastPredictions in
// tests/training-browser.mjs: a fake camera can't produce a real hand.
function testHands() {
  return typeof window !== "undefined" ? window.__sortieTestHands : null;
}
