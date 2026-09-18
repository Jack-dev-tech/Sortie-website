import test from "node:test";
import assert from "node:assert/strict";
import { personMask, addBox, dilateMask, privacyMask } from "../static/js/privacy-mask.mjs";
import { PrivacyCamera } from "../static/js/privacy.mjs";

test("person confidence is conservative and rejects invalid results", () => {
  assert.deepEqual([...personMask([0, 0.24, 0.25, 1], 2, 2)], [0, 0, 1, 1]);
  assert.throws(() => personMask([1], 2, 2));
  assert.throws(() => personMask([NaN], 1, 1));
});

test("padded face/hand regions clip at edges and retain separate people", () => {
  const mask = new Uint8Array(100);
  addBox(mask, 10, 10, { x: -0.1, y: -0.1, w: 0.3, h: 0.3 }, 0);
  addBox(mask, 10, 10, { x: 0.8, y: 0.8, w: 0.4, h: 0.4 }, 0);
  assert.equal(mask[0], 1);
  assert.equal(mask[99], 1);
  assert.equal(mask[55], 0);
  const before = mask.slice();
  addBox(mask, 10, 10, { x: -2, y: 0, w: 0.1, h: 1 });
  assert.deepEqual(mask, before);
  assert.throws(() => addBox(mask, 10, 10, { x: NaN, y: 0, w: 1, h: 1 }));
});

test("dilation covers boundary gaps without blurring the whole scene", () => {
  const mask = new Uint8Array(49);
  mask[24] = 1;
  const rgba = dilateMask(mask, 7, 7, 1);
  assert.equal(rgba.filter((v) => v === 255).length, 9);
  assert.equal(rgba[24 * 4 + 3], 255);
  assert.equal(rgba[3], 0);
  assert.equal(dilateMask(new Uint8Array(49), 7, 7).some(Boolean), false);
  mask.fill(1);
  assert.equal(dilateMask(mask, 7, 7).filter((v) => v === 255).length, 49);
});

test("hands and nearby held-item areas remain clear after body dilation", () => {
  const body = new Uint8Array(10000).fill(1);
  const pixels = privacyMask(body, 100, 100, [], [{ x: 0.4, y: 0.4, w: 0.2, h: 0.2 }]);
  const alpha = (x, y) => pixels[(y * 100 + x) * 4 + 3];
  assert.equal(alpha(50, 50), 0); // hand
  assert.equal(alpha(35, 50), 0); // nearby held item in the 30% padding
  assert.equal(alpha(32, 50), 255); // neighboring body stays blurred
  assert.equal(alpha(50, 32), 255);
});

test("face protection wins over overlapping hands, including expanded face boundaries", () => {
  const face = { x: 0.4, y: 0.4, w: 0.2, h: 0.2 };
  const faces = new Uint8Array(10000);
  addBox(faces, 100, 100, face, 0.25);
  const expectedFaces = dilateMask(faces, 100, 100);
  // A huge hand region would otherwise clear the entire frame.
  const result = privacyMask(new Uint8Array(10000).fill(1), 100, 100,
    [face], [{ x: 0, y: 0, w: 1, h: 1 }]);
  assert.deepEqual(result, expectedFaces);
  assert.deepEqual(privacyMask(new Uint8Array(10000), 100, 100,
    [face], [{ x: 0, y: 0, w: 1, h: 1 }]), expectedFaces);
});

test("multiple hands clip at frame edges while other body regions stay blurred", () => {
  const body = new Uint8Array(10000).fill(1);
  const result = privacyMask(body, 100, 100, [], [
    { x: -0.05, y: -0.05, w: 0.2, h: 0.2 },
    { x: 0.85, y: 0.85, w: 0.2, h: 0.2 },
  ]);
  assert.equal(result[3], 0);
  assert.equal(result[result.length - 1], 0);
  assert.equal(result[(50 * 100 + 50) * 4 + 3], 255);
});

test("missing hand detections leave the body blur intact", () => {
  const body = new Uint8Array(10000);
  addBox(body, 100, 100, { x: 0.2, y: 0.2, w: 0.6, h: 0.6 });
  assert.deepEqual(privacyMask(body, 100, 100), dilateMask(body, 100, 100));
});

function harness(t) {
  const scheduled = new Map(), workers = [], drawings = [], errors = [], frames = [];
  let nextRAF = 0;
  class Worker {
    constructor() { workers.push(this); this.sent = []; }
    postMessage(data) { this.sent.push(data); }
    terminate() { this.terminated = true; }
    receive(data) { this.onmessage({ data }); }
  }
  const globals = {
    Worker, OffscreenCanvas: class {}, document: { hidden: false },
    requestAnimationFrame(fn) { scheduled.set(++nextRAF, fn); return nextRAF; },
    cancelAnimationFrame(id) { scheduled.delete(id); },
    async createImageBitmap() { return { raw: true, close() { this.closed = true; } }; },
  };
  const restore = [];
  for (const [name, value] of Object.entries(globals)) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { value, writable: true, configurable: true });
    restore.push(() => original ? Object.defineProperty(globalThis, name, original) : delete globalThis[name]);
  }
  const video = { readyState: 2, currentTime: 1, videoWidth: 1280, videoHeight: 720 };
  const context = { drawImage: (bitmap) => drawings.push(bitmap), clearRect: () => drawings.length = 0 };
  const canvas = { width: 1, height: 1, getContext: () => context };
  const camera = new PrivacyCamera(video, canvas, {
    onFrame: () => frames.push(true), onError: (error) => errors.push(error),
  });
  t.after(() => { camera.stop(); restore.forEach((fn) => fn()); });
  async function capture() {
    const [id, fn] = scheduled.entries().next().value;
    scheduled.delete(id);
    await fn();
    await Promise.resolve();
  }
  const bitmap = () => ({ width: 640, height: 360, close() { this.closed = true; } });
  return { camera, video, workers, drawings, frames, errors, scheduled, capture, bitmap };
}

test("nothing renders until the matching processed frame arrives; only one frame is queued", async (t) => {
  const h = harness(t);
  h.camera.start();
  assert.equal(h.camera.available, false);
  assert.equal(h.drawings.length, 0);
  const worker = h.workers[0];
  worker.receive({ type: "ready" });
  await h.capture();
  assert.equal(h.scheduled.size, 0);
  assert.equal(h.drawings.length, 0);
  const frame = worker.sent[1];
  assert.equal(frame.bitmap.raw, true);
  const processed = h.bitmap();
  worker.receive({ type: "frame", id: frame.id, bitmap: processed });
  assert.deepEqual(h.drawings, [processed]);
  assert.equal(processed.closed, true);
  assert.equal(h.camera.available, true);
  assert.equal(h.frames.length, 1);
  assert.equal(h.scheduled.size, 1);
});

test("restart discards old worker frames and clears the preview", async (t) => {
  const h = harness(t);
  h.camera.start();
  const old = h.workers[0];
  old.receive({ type: "ready" });
  await h.capture();
  h.camera.start();
  const stale = h.bitmap();
  old.receive({ type: "frame", id: old.sent[1].id, bitmap: stale });
  assert.equal(old.terminated, true);
  assert.equal(stale.closed, true);
  assert.equal(h.camera.available, false);
  assert.equal(h.frames.length, 0);
  assert.equal(h.drawings.length, 0);
});

test("mismatched frames and worker failures fail closed", async (t) => {
  const h = harness(t);
  h.camera.start();
  const worker = h.workers[0];
  worker.receive({ type: "ready" });
  await h.capture();
  worker.receive({ type: "frame", id: -1, bitmap: h.bitmap() });
  assert.equal(h.errors[0].name, "PrivacyError");
  assert.equal(h.camera.available, false);
  assert.equal(h.drawings.length, 0);
  h.camera.start();
  h.workers[1].receive({ type: "error", message: "Model unavailable" });
  assert.equal(h.errors.length, 2);
  assert.equal(h.workers[1].terminated, true);
});

test("startup and frame timeouts stop processing instead of exposing raw video", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(t);
  h.camera.start();
  t.mock.timers.tick(30001);
  assert.equal(h.errors.length, 1);
  assert.equal(h.camera.available, false);
  h.camera.start();
  h.workers[1].receive({ type: "ready" });
  await h.capture();
  t.mock.timers.tick(10001);
  assert.equal(h.errors.length, 2);
  assert.equal(h.drawings.length, 0);
});

test("a capture resolving after stop is closed without being sent", async (t) => {
  const h = harness(t);
  let resolve;
  globalThis.createImageBitmap = () => new Promise((r) => { resolve = r; });
  h.camera.start();
  const worker = h.workers[0];
  worker.receive({ type: "ready" });
  const capturing = h.capture();
  h.camera.stop();
  const raw = h.bitmap();
  resolve(raw);
  await capturing;
  assert.equal(raw.closed, true);
  assert.equal(worker.sent.length, 1);
  assert.equal(h.drawings.length, 0);
});
