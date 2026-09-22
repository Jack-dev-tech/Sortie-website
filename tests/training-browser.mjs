import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SORTIE_TEST_URL || 'http://127.0.0.1:5055';
const browser = await chromium.launch({channel: 'chrome', headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});
const status = {configured: true, captured: 0, uploaded: 0, pending: 0, failed: 0,
  capacity: 500, full: false, can_capture: true, project_url: 'https://app.roboflow.com/example/project'};
try {
  const page = await browser.newPage({viewport: {width: 1280, height: 900}});
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.route('**/training/status', route => route.fulfill({json: status}));
  await page.route('**/training/retry', route => route.fulfill({json: status}));
  await page.route('**/training/captures', route => {
    status.captured++; status.pending++;
    return route.fulfill({status: 201, json: {id: route.request().postDataJSON().id}});
  });
  await page.addInitScript(() => {
    // A real hand can't appear in a synthetic camera stream, so stand in for the
    // hand tracker: one hand parked in the middle of the frame, normalised to
    // raw camera coords. app.js then gates on motion inside its reach, which
    // `window.testStill` still controls.
    window.__sortieTestHands = [{ x: 0.4, y: 0.4, w: 0.2, h: 0.2 }];

    // Fake camera: a canvas stream that alternates between two flat colors, so
    // motion is deterministic and `window.testStill` freezes the scene.
    const source = document.createElement('canvas');
    source.width = 640; source.height = 360;
    let i = 0;
    window.testShade = 17; // #111
    setInterval(() => {
      if (!window.testStill) i++;
      window.testShade = i % 2 ? 170 : 17;
      const ctx = source.getContext('2d');
      ctx.fillStyle = i % 2 ? '#aaa' : '#111';
      ctx.fillRect(0, 0, 640, 360);
    }, 280);
    navigator.mediaDevices.getUserMedia = async () => source.captureStream(15);

    // Average brightness of a JPEG data URL, for checking what was submitted.
    window.shadeOf = (dataUrl) => new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        const { data } = c.getContext('2d').getImageData(0, 0, img.width, img.height);
        let total = 0;
        for (let p = 0; p < data.length; p += 4) total += data[p];
        resolve({ width: img.width, shade: total / (data.length / 4) });
      };
      img.onerror = reject;
      img.src = dataUrl;
    });

    window.calls = [];
    window.checks = [];
    const nativeFetch = window.fetch;
    window.fetch = (url, options) => {
      if (url === '/predict' || url === '/training/captures') {
        const record = {url, at: Date.now(), mode: document.querySelector('.app').dataset.mode};
        if (url === '/training/captures') {
          const data = JSON.parse(options.body);
          const expected = window.testShade;
          // The capture is the live frame: JPEG, right size, right flat color.
          record.matches = data.image.startsWith('data:image/jpeg');
          window.checks.push(window.shadeOf(data.image).then(
            ({ width, shade }) => { record.matches = record.matches && width <= 640 && Math.abs(shade - expected) < 12; },
            () => { record.matches = false; }));
        }
        window.calls.push(record);
      }
      // Deliberately ignore abort to verify late prediction results are discarded.
      if (url === '/predict') return new Promise(resolve => {
        window.delayedPrediction = () => resolve(new Response(JSON.stringify({category:'plastic',confidence:.98})));
        if (window.fastPredictions) window.delayedPrediction();
      });
      return nativeFetch(url, options);
    };
  });
  await page.goto(base);
  await page.waitForFunction(() => window.delayedPrediction);
  await page.locator('button[data-mode="training"]').click();
  await page.evaluate(() => window.delayedPrediction());
  await page.waitForFunction(() => window.calls.filter(c => c.url === '/training/captures').length >= 3);
  await page.evaluate(() => Promise.all(window.checks));
  let captures = await page.evaluate(() => window.calls.filter(c => c.url === '/training/captures'));
  assert.ok(captures.every(c => c.matches && c.mode === 'training'));
  assert.ok(captures.slice(1).every((c, i) => c.at - captures[i].at >= 3000));
  assert.equal(await page.locator('.app').getAttribute('data-state'), 'idle');
  assert.equal(await page.evaluate(() => window.calls.some(c => c.url === '/predict' && c.mode === 'training')), false);
  // One red box, drawn over the fake hand rather than over the whole frame.
  assert.equal(await page.locator('.motion-box:not([hidden])').count(), 1);
  const [view, box] = await page.evaluate(() => [
    document.querySelector('[data-viewport]').getBoundingClientRect(),
    document.querySelector('.motion-box:not([hidden])').getBoundingClientRect(),
  ]);
  assert.ok(box.width > 0 && box.width < view.width && box.height < view.height);
  const count = () => page.evaluate(() => window.calls.filter(c => c.url === '/training/captures').length);
  await page.locator('[data-training-pause]').click();
  let before = await count(); await page.waitForTimeout(3400); assert.equal(await count(), before);
  await page.evaluate(() => window.testStill = true);
  await page.waitForTimeout(600);
  await page.locator('[data-training-pause]').click();
  await page.waitForTimeout(3400); assert.equal(await count(), before);
  await page.evaluate(() => window.testStill = false);
  await page.waitForFunction(n => window.calls.filter(c => c.url === '/training/captures').length > n, before);
  await page.evaluate(() => { Object.defineProperty(document, 'hidden', {value:true,configurable:true}); document.dispatchEvent(new Event('visibilitychange')); });
  before = await count(); await page.waitForTimeout(3400); assert.equal(await count(), before);
  await page.evaluate(() => { Object.defineProperty(document, 'hidden', {value:false,configurable:true}); document.dispatchEvent(new Event('visibilitychange')); });
  await page.waitForFunction(n => window.calls.filter(c => c.url === '/training/captures').length > n, before);
  await page.screenshot({path:'/tmp/sortie-training-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path:'/tmp/sortie-training-mobile.png', fullPage:true});
  await page.locator('button[data-mode="normal"]').focus();
  await page.keyboard.press('Enter');
  await page.evaluate(() => { window.fastPredictions = true; window.delayedPrediction?.(); });
  await page.waitForFunction(() => document.querySelector('.app').dataset.state === 'result');
  await page.locator('button[data-mode="training"]').click();
  assert.equal(await page.locator('.flyer').count(), 0);
  await page.waitForTimeout(4100);
  assert.equal(await page.locator('.app').getAttribute('data-state'), 'idle');
  assert.equal(await page.locator('.bins').isVisible(), false);
  assert.equal(await page.evaluate(() => window.calls.some(c => c.url === '/predict' && c.mode === 'training')), false);
  status.full = true; status.pending = 500; status.can_capture = false;
  await page.waitForFunction(() => document.querySelector('[data-training-status]').textContent.includes('Queue full'));
  before = await count(); await page.waitForTimeout(3400); assert.equal(await count(), before);
  status.configured = false; status.setup = 'Set ROBOFLOW_API_KEY on server'; status.project_url = null;
  await page.waitForFunction(() => document.querySelector('[data-training-status]').textContent.includes('ROBOFLOW_API_KEY'));
  assert.equal(await page.locator('[data-training-pause]').isDisabled(), true);
  await page.reload();
  assert.equal(await page.locator('.app').getAttribute('data-mode'), 'normal');
  assert.deepEqual(errors, []);
  console.log('PASS cadence, captured JPEG, still scenes, pause/resume, hidden tab, late predictions, verdict switches, queue full, setup, keyboard, mobile, reload');
} finally { await browser.close(); }
