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
  const FRAME_WIDTH = 320; // downscale width sent over the wire (keeps it light)
  const CONFIDENCE_MIN = 0.75; // ignore predictions less certain than this
  const STABLE_FRAMES = 2; // same category this many times in a row = locked
  const RESULT_HOLD = 3800; // ms the verdict stays up before scanning resumes

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
    glass:   { color: "--glass",   cheer: "Glass goes here — thanks!" },
    paper:   { color: "--paper",   cheer: "Paper, sorted. Nice one!" },
    plastic: { color: "--plastic", cheer: "Plastic in the recycling — great!" },
    waste:   { color: "--waste",   cheer: "Landfill it is. Every bit counts." },
  };

  // ---- Elements ---------------------------------------------------------
  const app = document.querySelector(".app");
  const feed = document.querySelector("[data-feed]");
  const grabber = document.querySelector("[data-grabber]");
  const viewport = document.querySelector("[data-viewport]");
  const verdictEl = document.querySelector("[data-verdict]");
  const binArrow = document.querySelector("[data-bin-arrow]");
  const promptEl = document.querySelector("[data-prompt]");
  const statusLabel = document.querySelector("[data-status-label]");
  const cheerEl = document.querySelector("[data-cheer]");
  const verdictName = document.querySelector("[data-verdict-name]");
  const verdictConf = document.querySelector("[data-verdict-conf]");
  const verdictIcon = document.querySelector("[data-verdict-icon]");
  const ring = document.querySelector("[data-ring]");
  const cameraError = document.querySelector("[data-camera-error]");
  const errorTitle = document.querySelector("[data-error-title]");
  const errorMsg = document.querySelector("[data-error-msg]");
  const retryBtn = document.querySelector("[data-retry]");
  const confettiCanvas = document.querySelector("[data-confetti]");
  const motionBoxes = document.querySelector("[data-motion-boxes]");
  const bins = new Map(
    [...document.querySelectorAll("[data-bin]")].map((el) => [el.dataset.bin, el])
  );

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const RING_CIRCUMFERENCE = 327;

  // ---- State machine ----------------------------------------------------
  const PROMPTS = {
    idle: "Show me your item",
    scanning: "Hold it steady…",
    result: "",
    error: "",
  };
  const STATUS = {
    loading: "Waking up…",
    idle: "Ready",
    scanning: "Scanning…",
    result: "Sorted!",
    error: "Camera off",
  };

  const training = new TrainingCapture();
  const handTracker = new HandTracker();
  let modeGeneration = 0;
  let confettiRaf = null;
  let state = "loading";
  let stream = null;
  let lastMotionAt = 0;
  let inFlight = false;
  let holding = false; // true while a verdict is on screen
  let candidate = null; // { category, count } for the stability check
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
    if (next in PROMPTS) promptEl.textContent = training.active ? "Collect images for labeling" : PROMPTS[next];
    if (next in STATUS) statusLabel.textContent = STATUS[next];
  }

  function accentColor(category) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(CATEGORIES[category].color)
      .trim();
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
    candidate = null;
    lastPredictAt = lastMotionAt = 0;
    bins.forEach((el) => el.classList.remove("is-match"));
    cheerEl.textContent = "";
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
      const data = await res.json();
      if (!training.active && generation === modeGeneration && session === cameraSession && res.ok && data.category) {
        handlePrediction(data);
      }
      // Non-OK responses (bad frame, model not wired) are ignored so the loop
      // keeps running; the server logs the detail.
    } catch (_) {
      // network hiccup — skip this frame, try again next tick
    } finally {
      if (generation === modeGeneration && session === cameraSession) {
        inFlight = false;
        predictionRequest = null;
      }
    }
  }

  // ---- Stability check --------------------------------------------------
  function handlePrediction({ category, confidence }) {
    if (training.active || holding || !(category in CATEGORIES)) return;

    if (confidence < CONFIDENCE_MIN) {
      candidate = null;
      return;
    }
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
  function showResult(category, confidence) {
    holding = true;
    const color = accentColor(category);
    document.documentElement.style.setProperty("--accent", color);

    verdictName.textContent = category;
    verdictConf.textContent = `${Math.round(confidence * 100)}% sure`;

    // clone the matching bin's icon into the verdict + prep the flyer
    const matchBin = bins.get(category);
    const iconSvg = matchBin.querySelector(".bin-icon svg");
    verdictIcon.innerHTML = "";
    verdictIcon.appendChild(iconSvg.cloneNode(true));

    // confidence ring
    ring.style.strokeDashoffset = String(
      RING_CIRCUMFERENCE * (1 - Math.min(Math.max(confidence, 0), 1))
    );

    cheerEl.textContent = CATEGORIES[category].cheer;

    // highlight the right bin, dim the rest (handled in CSS via data-state)
    bins.forEach((el) => el.classList.remove("is-match"));
    matchBin.classList.add("is-match");

    setState("result");
    pointArrowAt(matchBin);

    if (!reduceMotion) {
      flyToBin(iconSvg, matchBin, color);
      burstConfetti(color);
    }

    resultTimer = setTimeout(endResult, RESULT_HOLD);
  }

  function endResult() {
    bins.forEach((el) => el.classList.remove("is-match"));
    ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    cheerEl.textContent = "";
    holding = false;
    prevGray = null; // re-baseline motion so the resume doesn't self-trigger
    setState(stream ? "scanning" : "error");
  }

  // ---- Arrow pointing at the matching bin -------------------------------
  function pointArrowAt(matchBin) {
    const bin = matchBin.getBoundingClientRect();
    const arrow = binArrow.getBoundingClientRect();
    binArrow.style.left = `${bin.left + bin.width / 2}px`;
    binArrow.style.top = `${bin.top - arrow.height - 10}px`;
  }

  // ---- Flying item icon -------------------------------------------------
  function flyToBin(iconSvg, matchBin, color) {
    const start = verdictEl.getBoundingClientRect();
    const end = matchBin.querySelector(".bin-icon").getBoundingClientRect();

    const flyer = document.createElement("div");
    flyer.className = "flyer";
    flyer.style.setProperty("--accent", color);
    flyer.style.background = color;
    flyer.appendChild(iconSvg.cloneNode(true));

    const size = 56;
    const x0 = start.left + start.width / 2 - size / 2;
    const y0 = start.top + start.height / 2 - size / 2;
    const x1 = end.left + end.width / 2 - size / 2;
    const y1 = end.top + end.height / 2 - size / 2;

    flyer.style.left = `${x0}px`;
    flyer.style.top = `${y0}px`;
    document.body.appendChild(flyer);

    const arcLift = Math.min(140, Math.abs(y1 - y0) * 0.5 + 60);
    flyer
      .animate(
        [
          { transform: "translate(0,0) scale(0.6)", opacity: 0 },
          { transform: "translate(0,0) scale(1)", opacity: 1, offset: 0.15 },
          {
            transform: `translate(${(x1 - x0) * 0.5}px, ${-arcLift}px) scale(1.05)`,
            opacity: 1,
            offset: 0.55,
          },
          { transform: `translate(${x1 - x0}px, ${y1 - y0}px) scale(0.4)`, opacity: 0 },
        ],
        { duration: 900, easing: "cubic-bezier(0.4, 0, 0.2, 1)", fill: "forwards" }
      )
      .addEventListener("finish", () => flyer.remove());
  }

  // ---- Confetti ---------------------------------------------------------
  function burstConfetti(color) {
    const canvas = confettiCanvas;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    ctx.scale(dpr, dpr);

    const palette = [color, "#ffffff", "#0f1613"];
    const originX = window.innerWidth / 2;
    const originY = window.innerHeight * 0.42;
    const pieces = Array.from({ length: 90 }, () => {
      const angle = Math.random() * Math.PI * 2;
      const speed = 6 + Math.random() * 9;
      return {
        x: originX,
        y: originY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 4,
        size: 5 + Math.random() * 6,
        rot: Math.random() * Math.PI,
        vr: (Math.random() - 0.5) * 0.3,
        color: palette[(Math.random() * palette.length) | 0],
        life: 1,
      };
    });

    let raf;
    function frame() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      let alive = false;
      for (const p of pieces) {
        p.vy += 0.35; // gravity
        p.vx *= 0.99;
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        p.life -= 0.012;
        if (p.life <= 0 || p.y > window.innerHeight + 40) continue;
        alive = true;
        ctx.save();
        ctx.globalAlpha = Math.max(p.life, 0);
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        ctx.restore();
      }
      if (alive) {
        raf = confettiRaf = requestAnimationFrame(frame);
      } else {
        cancelAnimationFrame(raf);
        ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      }
    }
    frame();
  }

  function selectMode(active) {
    if (training.active === active) return;
    modeGeneration++;
    predictionRequest?.abort();
    predictionRequest = null;
    inFlight = false;
    clearTimeout(resultTimer);
    holding = false;
    candidate = null;
    prevGray = null;
    bins.forEach((el) => el.classList.remove("is-match"));
    document.querySelectorAll(".flyer").forEach((el) => {
      el.getAnimations().forEach((animation) => animation.cancel());
      el.remove();
    });
    cancelAnimationFrame(confettiRaf);
    confettiCanvas.getContext("2d").clearRect(0, 0, confettiCanvas.width, confettiCanvas.height);
    cheerEl.textContent = "";
    ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    document.documentElement.style.removeProperty("--accent");
    app.dataset.mode = active ? "training" : "normal";
    training.select(active);
    document.querySelectorAll("button[data-mode]").forEach((button) => {
      button.setAttribute("aria-pressed", String((button.dataset.mode === "training") === active));
    });
    setState(state === "error" ? "error" : cameraReady ? "idle" : "loading");
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
