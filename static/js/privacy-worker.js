// Classic worker: MediaPipe's WASM loader uses importScripts internally.
// Dynamic imports let the application modules remain ES modules.
let segmenter, faceDetector, handDetector, maskOps;
let output, maskCanvas, blurCanvas, tinyCanvas;
let ready = false;

async function initialize() {
  const { FilesetResolver, ImageSegmenter, FaceDetector, HandLandmarker } =
    await import("../vendor/mediapipe/vision_bundle.mjs");
  maskOps = await import("./privacy-mask.mjs");
  const files = await FilesetResolver.forVisionTasks(
    new URL("../vendor/mediapipe/wasm", self.location.href).href
  );
  const options = (name) => ({
    baseOptions: {
      modelAssetPath: new URL(`../models/privacy/${name}`, self.location.href).href,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
  });
  // Initialize sequentially: the WASM loader uses a shared ModuleFactory global.
  segmenter = await ImageSegmenter.createFromOptions(files, {
    ...options("selfie_segmenter.tflite"),
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });
  faceDetector = await FaceDetector.createFromOptions(files, {
    ...options("face_detector.tflite"), minDetectionConfidence: 0.35,
  });
  handDetector = await HandLandmarker.createFromOptions(files, {
    ...options("hand_landmarker.task"), numHands: 8,
    minHandDetectionConfidence: 0.35, minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.5,
  });
  output = new OffscreenCanvas(1, 1);
  maskCanvas = new OffscreenCanvas(1, 1);
  blurCanvas = new OffscreenCanvas(1, 1);
  tinyCanvas = new OffscreenCanvas(1, 1);
  if (!output.getContext("2d") || !maskCanvas.getContext("2d") ||
      !blurCanvas.getContext("2d") || !tinyCanvas.getContext("2d")) {
    throw new Error("Canvas processing unavailable");
  }
  ready = true;
  self.postMessage({ type: "ready" });
}

function processFrame(bitmap, timestamp, id) {
  const width = bitmap.width, height = bitmap.height;
  let mask, mw, mh;
  // Confidence masks belong to the callback; copy their data before it returns.
  segmenter.segmentForVideo(bitmap, timestamp, (result) => {
    const person = result.confidenceMasks?.[0];
    if (!person) throw new Error("Person segmentation unavailable");
    mw = person.width;
    mh = person.height;
    mask = maskOps.personMask(person.getAsFloat32Array(), mw, mh);
  });
  if (!mask) throw new Error("Missing person mask");
  const faces = faceDetector.detectForVideo(bitmap, timestamp).detections.map((face) => {
    const box = face.boundingBox;
    if (!box) throw new Error("Missing face bounds");
    return {
      x: box.originX / width, y: box.originY / height,
      w: box.width / width, h: box.height / height,
    };
  });
  const hands = handDetector.detectForVideo(bitmap, timestamp).landmarks.map((points) => {
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const x = Math.min(...xs), y = Math.min(...ys);
    return {
      x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y,
    };
  });

  maskCanvas.width = mw;
  maskCanvas.height = mh;
  maskCanvas.getContext("2d").putImageData(
    new ImageData(maskOps.privacyMask(mask, mw, mh, faces, hands), mw, mh), 0, 0
  );
  output.width = blurCanvas.width = width;
  output.height = blurCanvas.height = height;
  // Heavy downsampling removes fine detail even where canvas filter is absent.
  // Smoothly enlarge, then soften again for a blurred (rather than blocky) result.
  tinyCanvas.width = Math.max(1, Math.ceil(width / 32));
  tinyCanvas.height = Math.max(1, Math.ceil(height / 32));
  tinyCanvas.getContext("2d").drawImage(bitmap, 0, 0, tinyCanvas.width, tinyCanvas.height);
  const blur = blurCanvas.getContext("2d");
  blur.imageSmoothingEnabled = true;
  blur.imageSmoothingQuality = "high";
  blur.filter = "blur(12px)";
  // Extend the image beyond the canvas to avoid transparent edges from the filter.
  blur.drawImage(tinyCanvas, -32, -32, width + 64, height + 64);
  blur.filter = "none";
  blur.globalCompositeOperation = "destination-in";
  blur.imageSmoothingEnabled = false;
  blur.drawImage(maskCanvas, 0, 0, width, height);
  blur.globalCompositeOperation = "source-over";

  const ctx = output.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  // Remove protected source pixels completely before compositing the blur.
  ctx.globalCompositeOperation = "destination-out";
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(maskCanvas, 0, 0, width, height);
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(blurCanvas, 0, 0);
  const processed = output.transferToImageBitmap();
  self.postMessage({ type: "frame", bitmap: processed, id }, [processed]);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await initialize();
    else if (data.type === "frame") {
      if (!ready) throw new Error("Privacy filter is not ready");
      processFrame(data.bitmap, data.timestamp, data.id);
    }
  } catch (error) {
    ready = false;
    self.postMessage({ type: "error", message: error.message });
  } finally {
    data.bitmap?.close();
  }
};
