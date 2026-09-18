// Run against a local Flask server. See README for optional browser test setup.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.SORTIE_TEST_URL || "http://127.0.0.1:5055";
const fixture = await readFile(process.env.SORTIE_TEST_PHOTO || "/tmp/sortie-privacy-pose.jpg");
const handsFixture = await readFile(process.env.SORTIE_TEST_HANDS || "/tmp/sortie-privacy-hands.jpg");
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
});

async function pageWithCamera(photo = fixture) {
  const page = await browser.newPage();
  await page.route("**/__test/person.jpg", (route) => route.fulfill({ contentType: "image/jpeg", body: photo }));
  await page.route("**/predict", (route) => route.fulfill({
    json: { category: "plastic", confidence: 0.95 },
  }));
  await page.addInitScript(() => {
    window.__privacyFrames = 0;
    window.__submissions = [];
    window.__workers = [];
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        window.__workers.push(this);
        this.addEventListener("message", ({ data }) => {
          if (data.type === "frame") window.__privacyFrames++;
        });
      }
    };
    const nativeFetch = window.fetch;
    window.fetch = (url, options) => {
      if (url === "/predict") {
        const expected = document.createElement("canvas");
        const preview = document.querySelector("[data-private-feed]");
        expected.width = 320;
        expected.height = Math.round(preview.height * 320 / preview.width);
        expected.getContext("2d").drawImage(preview, 0, 0, expected.width, expected.height);
        window.__submissions.push({
          matchesPreview: JSON.parse(options.body).image === expected.toDataURL("image/jpeg", 0.7),
        });
      }
      return nativeFetch(url, options);
    };
    navigator.mediaDevices.getUserMedia = async () => {
      const image = new Image();
      image.src = "/__test/person.jpg";
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = Math.round(image.height * 640 / image.width);
      const context = canvas.getContext("2d");
      const draw = () => {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        // A moving object in the background triggers the recycling classifier.
        context.fillStyle = Math.floor(performance.now() / 300) % 2 ? "white" : "black";
        context.fillRect(0, 0, 100, 100);
      };
      draw();
      const interval = setInterval(draw, 70);
      const stream = canvas.captureStream(15);
      for (const track of stream.getTracks()) {
        const stop = track.stop.bind(track);
        track.stop = () => { clearInterval(interval); stop(); };
      }
      return stream;
    };
  });
  return page;
}

async function assertCovered(page) {
  const status = await page.evaluate(() => {
    const video = document.querySelector("video");
    const canvas = document.querySelector("[data-private-feed]");
    return {
      rawHidden: video.hidden && getComputedStyle(video).display === "none",
      empty: !canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data.some(Boolean),
      stopped: !video.srcObject,
    };
  });
  assert.deepEqual(status, { rawHidden: true, empty: true, stopped: true });
}

try {
  const page = await pageWithCamera();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(base);
  await page.waitForFunction(() => window.__privacyFrames > 2, null, { timeout: 45000 });
  assert.equal(await page.locator("video").isVisible(), false);
  await page.waitForFunction(() => window.__submissions.length >= 2, null, { timeout: 20000 });
  assert.equal(await page.evaluate(() => window.__submissions.every((s) => s.matchesPreview)), true);
  await page.waitForFunction(() => document.querySelector(".app").dataset.state === "result");
  const before = await page.evaluate(() => window.__privacyFrames);
  await page.waitForFunction((count) => window.__privacyFrames > count + 2, before);
  const previewPNG = await page.locator("[data-private-feed]").evaluate((canvas) => canvas.toDataURL("image/png"));
  await writeFile("/tmp/sortie-privacy-person.png", Buffer.from(previewPNG.split(",")[1], "base64"));
  // Quantify change in the subject versus an untouched background patch.
  const changes = await page.evaluate(async () => {
    const image = new Image(); image.src = "/__test/person.jpg"; await image.decode();
    const preview = document.querySelector("[data-private-feed]");
    const original = document.createElement("canvas");
    original.width = preview.width; original.height = preview.height;
    original.getContext("2d").drawImage(image, 0, 0, original.width, original.height);
    const raw = original.getContext("2d").getImageData(0, 0, original.width, original.height).data;
    const blurred = preview.getContext("2d").getImageData(0, 0, preview.width, preview.height).data;
    const difference = (x0, y0, x1, y1) => {
      let sum = 0, count = 0;
      for (let y = Math.floor(y0 * preview.height); y < y1 * preview.height; y++) {
        for (let x = Math.floor(x0 * preview.width); x < x1 * preview.width; x++) {
          const p = (y * preview.width + x) * 4;
          for (let c = 0; c < 3; c++) { sum += Math.abs(raw[p + c] - blurred[p + c]); count++; }
        }
      }
      return sum / count;
    };
    return { person: difference(0.45, 0.4, 0.53, 0.65), background: difference(0.8, 0.1, 0.9, 0.2) };
  });
  assert.ok(changes.person > 10, JSON.stringify(changes));
  assert.ok(changes.background < 5, JSON.stringify(changes));
  console.log("PASS real models, person blur, clear background, submitted JPEGs, verdict rendering", changes);

  await page.evaluate(() => window.__workers.at(-1).dispatchEvent(
    new MessageEvent("message", { data: { type: "error", message: "Injected failure" } })
  ));
  await page.waitForFunction(() => document.querySelector(".app").dataset.state === "error");
  await assertCovered(page);
  await page.locator("[data-retry]").click();
  await page.waitForFunction((count) => window.__privacyFrames > count + 3, before, { timeout: 45000 });
  await page.waitForFunction(() => ["idle", "scanning", "result"].includes(document.querySelector(".app").dataset.state));
  // Simulate visibility transitions deterministically; native events use the same handler.
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { value: true, configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await assertCovered(page);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { value: false, configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.waitForFunction(() => ["idle", "scanning", "result"].includes(document.querySelector(".app").dataset.state), null, { timeout: 45000 });
  assert.deepEqual(pageErrors, []);
  console.log("PASS worker failure, retry, and tab pause/resume");
  await page.close();

  const hands = await pageWithCamera(handsFixture);
  await hands.goto(base);
  await hands.waitForFunction(() => window.__privacyFrames > 2 && window.__submissions.length > 0,
    null, { timeout: 45000 });
  const handChanges = await hands.evaluate(async () => {
    const image = new Image(); image.src = "/__test/person.jpg"; await image.decode();
    const preview = document.querySelector("[data-private-feed]");
    const original = document.createElement("canvas");
    original.width = preview.width; original.height = preview.height;
    original.getContext("2d").drawImage(image, 0, 0, original.width, original.height);
    const raw = original.getContext("2d").getImageData(0, 0, original.width, original.height).data;
    const processed = preview.getContext("2d").getImageData(0, 0, preview.width, preview.height).data;
    return [[0.17, 0.25, 0.25, 0.45], [0.68, 0.6, 0.8, 0.8]].map(([x0, y0, x1, y1]) => {
      let sum = 0, count = 0;
      for (let y = Math.floor(y0 * preview.height); y < y1 * preview.height; y++) {
        for (let x = Math.floor(x0 * preview.width); x < x1 * preview.width; x++) {
          const p = (y * preview.width + x) * 4;
          for (let c = 0; c < 3; c++) { sum += Math.abs(raw[p + c] - processed[p + c]); count++; }
        }
      }
      return sum / count;
    });
  });
  assert.ok(handChanges.every((difference) => difference < 2), JSON.stringify(handChanges));
  assert.equal(await hands.evaluate(() => window.__submissions.every((s) => s.matchesPreview)), true);
  console.log("PASS real hand detections stay clear in preview and submitted images", handChanges);
  await hands.close();

  const missing = await pageWithCamera();
  await missing.route("**/models/privacy/**", (route) => route.abort());
  await missing.goto(base);
  await missing.waitForFunction(() => document.querySelector(".app").dataset.state === "error", null, { timeout: 45000 });
  await assertCovered(missing);
  assert.equal(await missing.evaluate(() => window.__submissions.length), 0);
  console.log("PASS missing model assets: camera covered and no submissions");
  await missing.close();

  const denied = await pageWithCamera();
  await denied.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException("Denied", "NotAllowedError"); };
  });
  await denied.goto(base);
  await denied.waitForFunction(() => document.querySelector(".app").dataset.state === "error");
  await assertCovered(denied);
  assert.equal(await denied.locator("[data-error-title]").textContent(), "Camera access blocked");
  console.log("PASS camera denial");
  await denied.close();
} finally {
  await browser.close();
}
