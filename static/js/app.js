/* =========================================================================
   Sortie — client logic
   Webcam → sample frames → POST /predict → state machine → verdict animation
   ========================================================================= */

import { TrainingCapture } from "./training.mjs";
import { HandTracker } from "./hands.mjs";

(() => {
  "use strict";

  // ---- Tunables ---------------------------------------------------------
  const TRACK_INTERVAL = 100; // ms between hand/motion checks (the red box follows a hand)
  const PREDICT_MIN_GAP = 1200; // ms — don't fire the model more often than this
  const FRAME_WIDTH = 640; // width sent over the wire — the model resizes to 640 anyway
  const STABLE_FRAMES = 2; // same category this many times in a row = locked
  const RESULT_HOLD = 3800; // ms the verdict stays up before scanning resumes
  const IDLE_AFTER = 1500; // ms with no hand in frame before the kiosk goes back to idle
  const UNSURE_AFTER = 3; // unusable predictions in a row before asking the person to adjust
  const OFFLINE_AFTER = 3; // failed /predict calls in a row before saying sorting is offline

  // Motion detection: compare small greyscale snapshots frame-to-frame.
  const MOTION_WIDTH = 64; // downscaled width used for the frame-diff
  const MOTION_PIXEL_DELTA = 26; // per-pixel brightness change that counts as movement
  const MOTION_AREA_MIN = 0.01; // whole-frame fraction that must move — fallback path only

  // Hand tracking: how far past the hand to look for the thing it is holding,
  // and how much of that region has to change to call the held item "moving".
  const HAND_REACH = 0.9; // multiples of the hand box's larger side
  const MOTION_AREA_MIN_LOCAL = 0.02; // changed fraction *within reach*
  const MAX_BOXES = 2; // one red box per tracked hand

  // Training captures: bigger and less compressed than the classifier frame.
  const CAPTURE_WIDTH = 640;
  const CAPTURE_QUALITY = 0.95;

  const CATEGORIES = {
    glass:   { tip: "Rinse it if you can, then drop it in." },
    paper:   { tip: "Flatten boxes so they fit." },
    plastic: { tip: "Empty it first, then drop it in." },
    waste:   { tip: "This one can't be recycled here." },
  };

  // ---- Elements ---------------------------------------------------------
  const app = document.querySelector(".app");
  const feed = document.querySelector("[data-feed]");
  const grabber = document.querySelector("[data-grabber]");
  const viewport = document.querySelector("[data-viewport]");
  const verdictEl = document.querySelector("[data-verdict]");
  const promptEl = document.querySelector("[data-prompt]");
  const statusLabel = document.querySelector("[data-status-label]");
  const cheerEl = document.querySelector("[data-cheer]");
  const verdictName = document.querySelector("[data-verdict-name]");
  const verdictConf = document.querySelector("[data-verdict-conf]");
  const cameraError = document.querySelector("[data-camera-error]");
  const errorTitle = document.querySelector("[data-error-title]");
  const errorMsg = document.querySelector("[data-error-msg]");
  const retryBtn = document.querySelector("[data-retry]");
  const motionBoxes = document.querySelector("[data-motion-boxes]");
  const viewportLabel = document.querySelector("[data-viewport-label]");
  const statusEl = document.querySelector(".status");
  const bins = new Map(
    [...document.querySelectorAll("[data-bin]")].map((el) => [el.dataset.bin, el])
  );

  // Ignore predictions less certain than this. The server owns the value so one
  // env var tunes it:  SORTIE_CONFIDENCE_MIN=0.3 python app.py
  const CONFIDENCE_MIN = Number(app.dataset.confidenceMin) || 0.45;

  // Training is for whoever runs the kiosk, not the public: the mode switch only
  // appears at /?operator. Every load still starts in Normal.
  const operator = new URLSearchParams(location.search).has("operator");
  if (operator) {
    app.dataset.operator = "";
    document.querySelector(".mode-switch").hidden = false;
  }

  // ---- State machine ----------------------------------------------------
  const PROMPTS = {
    loading: "Starting the camera…",
    idle: "Hold your item up to the camera",
    scanning: "Hold it still…",
    unsure: "Turn it so the label faces the camera",
    result: "",
    offline: "Can't sort right now. Use the signs on each bin.",
    error: "Camera off. Use the signs on each bin.",
  };
  // Nothing detected at all reads differently from "detected, but not sure".
  const UNSURE_NOTHING = "Hold it closer to the camera";
  const STATUS = {
    loading: "Starting camera",
    idle: "Ready",
    scanning: "Looking",
    unsure: "Not sure",
    result: "Sorted",
    offline: "Sorting offline",
    error: "Camera off",
  };
  // Shown inside the camera box when it has no useful picture of its own.
  const VIEWPORT_LABELS = {
    loading: "Camera starting",
    offline: "Sorting offline",
  };

  const training = new TrainingCapture();
  const handTracker = new HandTracker();
  let modeGeneration = 0;
  let state = "loading";
  let stream = null;
  let lastMotionAt = 0;
  let inFlight = false;
  let holding = false; // true while a verdict is on screen
  let candidate = null; // { category, count } for the stability check
  let lowCount = 0; // unusable predictions in a row (too unsure, or nothing seen)
  let lowReason = null; // "unsure" | "nothing" — what the last unusable one was
  let failCount = 0; // failed /predict calls in a row
  let lastPresentAt = 0; // last time a hand (or motion, without tracking) was in frame
  let prevGray = null; // greyscale snapshot of the previous frame, for the diff
  let motionCanvas = null; // small offscreen canvas the diff is computed on
  let lastPredictAt = 0; // timestamp of the last frame handed to the model
  let cameraSession = 0;
  let predictionRequest = null;
  let resultTimer = null;
  let resumeCamera = false;
  let frameRaf = null;
  let cameraReady = false; // the video is playing and has real dimensions

  // Drives the capture loop off the live video, throttled to TRACK_INTERVAL.
  function frameLoop() {
    frameRaf = requestAnimationFrame(frameLoop);
    if (!stream || document.hidden || feed.readyState < 2 || !feed.videoWidth) return;
    cameraReady = true;
    if (state === "loading") setState("idle");
    if (Date.now() - lastMotionAt < TRACK_INTERVAL) return;
    lastMotionAt = Date.now();
    tick();
  }

  function setState(next) {
    state = next;
    app.dataset.state = next;
    let prompt = PROMPTS[next];
    if (next === "unsure" && lowReason === "nothing") prompt = UNSURE_NOTHING;
    if (training.active && next !== "loading" && next !== "error") prompt = "Collect images for labeling";
    if (next in PROMPTS) promptEl.textContent = prompt;
    if (next in STATUS) statusLabel.textContent = STATUS[next];
    viewportLabel.textContent = VIEWPORT_LABELS[next] || "";
  }

  function resetAttempts() {
    candidate = null;
    lowCount = 0;
    lowReason = null;
  }

  // ---- Camera -----------------------------------------------------------
  async function startCamera() {
    stopCamera();
    const session = cameraSession;
    hideError();
    setState("loading");
    try {
      const cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          // Halve the capture rate: most webcams default to ~30fps, so ask for 15.
          frameRate: { ideal: 15, max: 15 },
        },
        audio: false,
      });
      if (session !== cameraSession) {
        cameraStream.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = cameraStream;
      stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => {
        if (session === cameraSession) showError(new Error("Camera disconnected"));
      }));
      feed.srcObject = stream;
      await feed.play();
      if (session !== cameraSession) return;
      if (!frameRaf) frameLoop();
    } catch (err) {
      if (session === cameraSession) showError(err);
    }
  }

  function stopCamera() {
    cameraSession++;
    cancelAnimationFrame(frameRaf);
    frameRaf = null;
    cameraReady = false;
    training.ready = false;
    training.render();
    predictionRequest?.abort();
    predictionRequest = null;
    inFlight = false;
    clearTimeout(resultTimer);
    holding = false;
    resetAttempts();
    failCount = 0;
    lastPredictAt = lastMotionAt = lastPresentAt = 0;
    clearVerdict();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    feed.srcObject = null;
    hideMotionBoxes();
    prevGray = null;
  }

  function showError(err) {
    stopCamera();
    setState("error");
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    const missing = err && (err.name === "NotFoundError" || err.name === "OverconstrainedError");
    if (denied) {
      errorTitle.textContent = "Camera access blocked";
      errorMsg.textContent =
        "Sortie needs your camera to see items. Allow it in your browser, then try again.";
    } else if (missing) {
      errorTitle.textContent = "No camera found";
      errorMsg.textContent = "Connect a camera, then try again.";
    } else {
      errorTitle.textContent = "Camera is off";
      errorMsg.textContent = "Something interrupted the camera. Try turning it back on.";
    }
    cameraError.hidden = false;
  }

  function hideError() {
    cameraError.hidden = true;
  }

  // ---- Capture loop -----------------------------------------------------
  // The loop watches for a hand; the model only runs while one is holding
  // something up. Movement elsewhere in the room is ignored.
  function tick() {
    if (!stream || !cameraReady || document.hidden) return;

    const diff = diffFrame();
    const hands = handTracker.detect(feed);
    const regions = handTracker.available ? handRegions(hands, diff) : fallbackRegions(diff);
    drawMotionBoxes(regions);

    // With hand tracking a hand in frame is the trigger; without it, any motion is.
    const present = handTracker.available ? hands.length > 0 : regions.length > 0;

    if (training.active) {
      // Training also waits for the item to move, so someone holding still
      // doesn't fill Label Studio with near-identical frames.
      const itemMoved = regions.some((region) => region.moved);
      training.tick(present && itemMoved, () => captureJPEG(CAPTURE_WIDTH, CAPTURE_QUALITY), true);
      return;
    }

    // A verdict is on screen: keep the box tracking the hand, but don't
    // re-classify until it clears.
    if (holding) return;

    // Once the person walks away, drop back to the invitation. Offline stays
    // up until a prediction actually succeeds.
    if (present) {
      lastPresentAt = Date.now();
    } else if ((state === "scanning" || state === "unsure") && Date.now() - lastPresentAt > IDLE_AFTER) {
      resetAttempts();
      setState("idle");
    }

    // An item held perfectly steady still gets classified — only an empty
    // frame stops the model.
    if (!present || inFlight) return;
    if (Date.now() - lastPredictAt < PREDICT_MIN_GAP) return;

    const frame = captureJPEG(FRAME_WIDTH, 0.7);
    if (frame) {
      lastPredictAt = Date.now();
      sendFrame(frame);
    }
  }

  // ---- Motion detection -------------------------------------------------
  // Diff a small greyscale version of the current frame against the last one.
  // Returns the raw per-cell change mask; callers pick which part of it matters.
  function diffFrame() {
    const vw = feed.videoWidth;
    const vh = feed.videoHeight;
    if (!vw || !vh) return null;

    if (!motionCanvas) motionCanvas = document.createElement("canvas");
    const gw = MOTION_WIDTH;
    const gh = Math.max(1, Math.round((vh / vw) * gw));
    motionCanvas.width = gw;
    motionCanvas.height = gh;

    const ctx = motionCanvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(feed, 0, 0, gw, gh);
    const { data } = ctx.getImageData(0, 0, gw, gh);

    const gray = new Uint8ClampedArray(gw * gh);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      // Rec. 601 luma — cheap and good enough for a difference check.
      gray[p] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    // First frame after (re)start: nothing to compare against yet.
    if (!prevGray || prevGray.length !== gray.length) {
      prevGray = gray;
      return null;
    }

    const changed = new Uint8Array(gw * gh);
    for (let p = 0; p < gray.length; p++) {
      if (Math.abs(gray[p] - prevGray[p]) > MOTION_PIXEL_DELTA) changed[p] = 1;
    }
    prevGray = gray;
    return { w: gw, h: gh, changed };
  }

  const cell = (v, n) => Math.min(n - 1, Math.max(0, Math.floor(v)));

  // Bounding box of the cells that changed inside `region` (a normalised box, or
  // the whole frame when null), plus how much of that region moved.
  function boxOfChanged(diff, region) {
    if (!diff) return null;
    const { w: gw, h: gh, changed } = diff;
    const x0 = region ? cell(region.x * gw, gw) : 0;
    const y0 = region ? cell(region.y * gh, gh) : 0;
    const x1 = region ? cell(Math.ceil((region.x + region.w) * gw) - 1, gw) : gw - 1;
    const y1 = region ? cell(Math.ceil((region.y + region.h) * gh) - 1, gh) : gh - 1;

    let moved = 0;
    let minX = gw, minY = gh, maxX = -1, maxY = -1;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (!changed[y * gw + x]) continue;
        moved++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    if (maxX < 0) return null;

    // Pad the box out by one cell so it hugs the object a little less tightly.
    minX = Math.max(0, minX - 1);
    minY = Math.max(0, minY - 1);
    maxX = Math.min(gw - 1, maxX + 1);
    maxY = Math.min(gh - 1, maxY + 1);

    return {
      box: {
        x: minX / gw,
        y: minY / gh,
        w: (maxX - minX + 1) / gw,
        h: (maxY - minY + 1) / gh,
      },
      moved,
      cells: (x1 - x0 + 1) * (y1 - y0 + 1),
    };
  }

  // ---- Regions of interest ----------------------------------------------
  // What the red box tracks: a hand, plus whatever is moving within its reach —
  // the item it is holding. Motion anywhere else falls outside every reach box
  // and is ignored, so a passer-by or a shifting shadow never widens the box.
  function handRegions(hands, diff) {
    const regions = hands.slice(0, MAX_BOXES).map((hand) => {
      const reach = grow(hand, Math.max(hand.w, hand.h) * HAND_REACH);
      const item = boxOfChanged(diff, reach);
      const moved = Boolean(item) && item.moved / item.cells >= MOTION_AREA_MIN_LOCAL;
      return { ...(item ? union(hand, item.box) : hand), moved };
    });
    // Two hands on the same item read as one box rather than two crossing ones.
    if (regions.length === 2 && overlaps(regions[0], regions[1])) {
      return [{ ...union(regions[0], regions[1]), moved: regions[0].moved || regions[1].moved }];
    }
    return regions;
  }

  // Without hand tracking the box is whatever moved anywhere in frame — exactly
  // how Sortie behaved before hand detection existed.
  function fallbackRegions(diff) {
    const found = boxOfChanged(diff, null);
    if (!found || found.moved / found.cells < MOTION_AREA_MIN) return [];
    return [{ ...found.box, moved: true }];
  }

  function grow(box, by) {
    const x = Math.max(0, box.x - by);
    const y = Math.max(0, box.y - by);
    return {
      x,
      y,
      w: Math.min(1, box.x + box.w + by) - x,
      h: Math.min(1, box.y + box.h + by) - y,
    };
  }

  function union(a, b) {
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
  }

  const overlaps = (a, b) =>
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

  // ---- Red boxes --------------------------------------------------------
  // Map normalised boxes from raw camera space into the mirrored, object-fit:
  // cover viewport and position one overlay over each.
  function drawMotionBoxes(regions) {
    const cw = viewport.clientWidth;
    const ch = viewport.clientHeight;
    const vw = feed.videoWidth;
    const vh = feed.videoHeight;
    if (!vw || !vh || !cw || !ch) return;

    const scale = Math.max(cw / vw, ch / vh); // object-fit: cover
    const dw = vw * scale;
    const dh = vh * scale;
    const offX = (dw - cw) / 2;
    const offY = (dh - ch) / 2;

    regions.forEach((box, i) => {
      const el = boxAt(i);
      const w = box.w * dw;
      const h = box.h * dh;
      const top = box.y * dh - offY;
      // The feed is mirrored (scaleX(-1)), so flip the box horizontally too.
      const left = cw - (box.x * dw - offX + w);

      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      el.style.width = `${w}px`;
      el.style.height = `${h}px`;
      el.hidden = false;
    });
    for (let i = regions.length; i < motionBoxes.children.length; i++) {
      motionBoxes.children[i].hidden = true;
    }
  }

  // The overlay pool — grows to at most one element per tracked hand.
  function boxAt(i) {
    while (motionBoxes.children.length <= i) {
      const el = document.createElement("div");
      el.className = "motion-box";
      motionBoxes.appendChild(el);
    }
    return motionBoxes.children[i];
  }

  function hideMotionBoxes() {
    for (const el of motionBoxes.children) el.hidden = true;
  }

  // Snapshot the live video into the offscreen grabber canvas as a JPEG data URL.
  function captureJPEG(maxWidth, quality) {
    const vw = feed.videoWidth;
    const vh = feed.videoHeight;
    if (!cameraReady || !vw || !vh) return null;
    const width = Math.min(maxWidth, vw);
    grabber.width = width;
    grabber.height = Math.max(1, Math.round((vh * width) / vw));
    const ctx = grabber.getContext("2d");
    ctx.drawImage(feed, 0, 0, grabber.width, grabber.height);
    return grabber.toDataURL("image/jpeg", quality);
  }

  async function sendFrame(dataUrl) {
    if (training.active) return;
    const generation = modeGeneration;
    const session = cameraSession;
    const controller = new AbortController();
    predictionRequest = controller;
    inFlight = true;
    if (state === "idle") setState("scanning");
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
        signal: controller.signal,
      });
      const data = await res.json().catch(() => ({}));
      if (training.active || generation !== modeGeneration || session !== cameraSession) return;
      if (res.ok) {
        predictionSucceeded();
        handlePrediction(data);
      } else {
        // Bad frame, model not wired, model crashed: the server logs the detail.
        predictionFailed(data.error || `HTTP ${res.status}`);
      }
    } catch (err) {
      // Network hiccup, or the server is down. Aborts are deliberate, not failures.
      if (err.name !== "AbortError" && generation === modeGeneration && session === cameraSession) {
        predictionFailed(err.message);
      }
    } finally {
      if (generation === modeGeneration && session === cameraSession) {
        inFlight = false;
        predictionRequest = null;
      }
    }
  }

  // ---- Server health ----------------------------------------------------
  // A few failures in a row means sorting is down, not one dropped frame. The
  // loop keeps asking at the normal pace, so it recovers on its own.
  function predictionFailed(detail) {
    failCount += 1;
    // Operators can hover the status pill for the reason; the public never sees it.
    if (operator) statusEl.title = `Last /predict error: ${detail}`;
    if (failCount >= OFFLINE_AFTER && !holding && state !== "offline") {
      resetAttempts();
      setState("offline");
    }
  }

  function predictionSucceeded() {
    failCount = 0;
    if (operator) statusEl.title = "Live camera status";
    if (state === "offline") setState("scanning");
  }

  // ---- Stability check --------------------------------------------------
  function handlePrediction({ category, confidence }) {
    if (training.active || holding) return;

    // Nothing recognised, or not sure enough: after a few in a row, tell the
    // person what to change instead of looking forever.
    const usable = category in CATEGORIES && confidence >= CONFIDENCE_MIN;
    if (!usable) {
      candidate = null;
      lowCount += 1;
      lowReason = category ? "unsure" : "nothing";
      if (lowCount >= UNSURE_AFTER) setState("unsure");
      return;
    }
    lowCount = 0;
    if (state === "unsure") setState("scanning");
    if (candidate && candidate.category === category) {
      candidate.count += 1;
    } else {
      candidate = { category, count: 1 };
    }
    if (candidate.count >= STABLE_FRAMES) {
      showResult(category, confidence);
      candidate = null;
    }
  }

  // ---- Verdict ----------------------------------------------------------
  // The matching plate's colour rises out of its bin (CSS clip-path from --ox).
  function showResult(category, confidence) {
    holding = true;
    resetAttempts();
    const matchBin = bins.get(category);
    const bin = matchBin.getBoundingClientRect();
    const verdict = verdictEl.getBoundingClientRect();
    verdictEl.style.setProperty("--ox", `${bin.left + bin.width / 2 - verdict.left}px`);

    app.dataset.category = category;
    verdictName.textContent = category;
    verdictConf.textContent = `${Math.round(confidence * 100)}% sure`;
    cheerEl.textContent = CATEGORIES[category].tip;

    bins.forEach((el) => el.classList.toggle("is-match", el === matchBin));
    setState("result");
    resultTimer = setTimeout(endResult, RESULT_HOLD);
  }

  function endResult() {
    holding = false;
    prevGray = null; // re-baseline motion so the resume doesn't self-trigger
    setState(stream ? "scanning" : "error");
    // Keep the colour and words until the flood has drained back into the bin.
    resultTimer = setTimeout(() => {
      if (state !== "result") clearVerdict();
    }, 450);
  }

  function clearVerdict() {
    bins.forEach((el) => el.classList.remove("is-match"));
    delete app.dataset.category;
    cheerEl.textContent = "";
  }

  function selectMode(active) {
    if (training.active === active) return;
    modeGeneration++;
    predictionRequest?.abort();
    predictionRequest = null;
    inFlight = false;
    clearTimeout(resultTimer);
    holding = false;
    resetAttempts();
    prevGray = null;
    app.dataset.mode = active ? "training" : "normal";
    training.select(active);
    document.querySelectorAll("button[data-mode]").forEach((button) => {
      button.setAttribute("aria-pressed", String((button.dataset.mode === "training") === active));
    });
    setState(state === "error" ? "error" : cameraReady ? "idle" : "loading");
    clearVerdict();
  }
  document.querySelectorAll("button[data-mode]").forEach((button) => {
    button.addEventListener("click", () => selectMode(button.dataset.mode === "training"));
  });

  // ---- Lifecycle --------------------------------------------------------
  retryBtn.addEventListener("click", startCamera);
  document.addEventListener("visibilitychange", () => {
    // Release raw frames and discard all pending work when leaving the tab.
    if (document.hidden) {
      resumeCamera = Boolean(stream) || state === "loading";
      stopCamera();
    } else if (resumeCamera) {
      resumeCamera = false;
      startCamera();
    }
  });
  window.addEventListener("pagehide", stopCamera);
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && !document.hidden) startCamera();
  });

  handTracker.load(); // resolves in the background; the loop falls back until it does
  startCamera();
})();
