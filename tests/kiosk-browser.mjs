import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SORTIE_TEST_URL || 'http://127.0.0.1:5055';
const browser = await chromium.launch({channel: 'chrome', headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});

// What /predict answers next; each step of the test swaps it.
let reply = {status: 200, json: {category: null, confidence: 0}};

const fakeCamera = () => {
  // Same stand-ins as training-browser.mjs: a hand parked mid-frame and a
  // flat-colour canvas stream that keeps changing.
  window.__sortieTestHands = [{ x: 0.4, y: 0.4, w: 0.2, h: 0.2 }];
  const source = document.createElement('canvas');
  source.width = 640; source.height = 360;
  let i = 0;
  setInterval(() => {
    const ctx = source.getContext('2d');
    ctx.fillStyle = ++i % 2 ? '#aaa' : '#111';
    ctx.fillRect(0, 0, 640, 360);
  }, 280);
  navigator.mediaDevices.getUserMedia = async () => source.captureStream(15);
};

const stateIs = (page, s) => page.waitForFunction(s => document.querySelector('.app').dataset.state === s, s,
  {timeout: 15000});
const prompt = page => page.locator('[data-prompt]').textContent();

try {
  const page = await browser.newPage({viewport: {width: 1280, height: 800}});
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.route('**/predict', route => route.fulfill(reply));
  await page.addInitScript(fakeCamera);
  await page.goto(base);

  // Public kiosk: no way into Training.
  assert.equal(await page.locator('.mode-switch').isVisible(), false);

  // Nothing recognised, three times running → ask them to bring it closer.
  await stateIs(page, 'unsure');
  assert.equal(await prompt(page), 'Hold it closer to the camera');

  // Recognised but not sure enough → ask them to turn it.
  reply = {status: 200, json: {category: 'plastic', confidence: 0.1}};
  await page.waitForFunction(() => document.querySelector('[data-prompt]').textContent.startsWith('Turn it'),
    null, {timeout: 15000});

  // The person walks away → back to the invitation.
  await page.evaluate(() => { window.__sortieTestHands = []; });
  await stateIs(page, 'idle');
  assert.equal(await prompt(page), 'Hold your item up to the camera');

  // The model is down → say so, and point at the bin signs.
  reply = {status: 501, json: {error: 'No model loaded'}};
  await page.evaluate(() => { window.__sortieTestHands = [{ x: 0.4, y: 0.4, w: 0.2, h: 0.2 }]; });
  await stateIs(page, 'offline');
  assert.match(await prompt(page), /Use the signs on each bin/);
  assert.equal(await page.locator('[data-viewport-label]').isVisible(), true);
  assert.equal(await page.locator('.status').getAttribute('title'), 'Live camera status'); // no detail for the public

  // It recovers by itself once predictions work again.
  reply = {status: 200, json: {category: 'glass', confidence: 0.95}};
  await stateIs(page, 'result');
  assert.equal(await page.locator('[data-verdict-name]').textContent(), 'glass');

  // Operators get the failure reason on the status pill.
  reply = {status: 501, json: {error: 'No model loaded'}};
  await page.goto(`${base}/?operator`);
  await stateIs(page, 'offline');
  assert.match(await page.locator('.status').getAttribute('title'), /No model loaded/);

  // Camera refused → the stage still says something useful.
  const denied = await browser.newPage();
  denied.on('pageerror', err => errors.push(err.message));
  await denied.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('no', 'NotAllowedError'); };
  });
  await denied.goto(base);
  await stateIs(denied, 'error');
  assert.match(await prompt(denied), /Camera off/);
  assert.equal(await denied.locator('[data-error-title]').textContent(), 'Camera access blocked');

  assert.deepEqual(errors, []);
  console.log('PASS public mode switch hidden, unsure (nothing / low confidence), idle return, offline + recovery, operator detail, camera denied');
} finally { await browser.close(); }
