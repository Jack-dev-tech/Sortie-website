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
  await page.route('**/static/js/privacy.mjs', route => route.fulfill({contentType: 'text/javascript', body: `
    export class PrivacyCamera {
      constructor(video, canvas, options) { this.canvas = canvas; this.options = options; window.testPrivacy = this; }
      start() { this.stop(); this.canvas.width = 640; this.canvas.height = 360;
        this.available = true; this.i = 0;
        this.timer = setInterval(() => {
          if (!window.testStill) this.i++;
          const ctx = this.canvas.getContext('2d');
          ctx.fillStyle = this.i % 2 ? '#aaa' : '#111'; ctx.fillRect(0, 0, 640, 360);
          this.options.onFrame();
        }, 280);
      }
      stop() { this.available = false; clearInterval(this.timer); this.canvas.getContext('2d').clearRect(0,0,640,360); }
    }` }));
  await page.addInitScript(() => {
    window.calls = [];
    const nativeFetch = window.fetch;
    window.fetch = (url, options) => {
      if (url === '/predict' || url === '/training/captures') {
        const record = {url, at: Date.now(), mode: document.querySelector('.app').dataset.mode};
        if (url === '/training/captures') {
          const data = JSON.parse(options.body);
          record.matches = data.image === document.querySelector('[data-private-feed]').toDataURL('image/jpeg', .95);
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
  let captures = await page.evaluate(() => window.calls.filter(c => c.url === '/training/captures'));
  assert.ok(captures.every(c => c.matches && c.mode === 'training'));
  assert.ok(captures.slice(1).every((c, i) => c.at - captures[i].at >= 3000));
  assert.equal(await page.locator('.app').getAttribute('data-state'), 'idle');
  assert.equal(await page.evaluate(() => window.calls.some(c => c.url === '/predict' && c.mode === 'training')), false);
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
  await page.waitForFunction(() => window.testPrivacy.available);
  await page.evaluate(() => window.testPrivacy.options.onError({name:'PrivacyError'}));
  before = await count(); await page.waitForTimeout(3400); assert.equal(await count(), before);
  assert.equal(await page.locator('[data-error-title]').textContent(), 'Privacy blur unavailable');
  await page.locator('[data-retry]').click();
  await page.waitForFunction(() => window.testPrivacy.available);
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
  console.log('PASS cadence, processed JPEG, still scenes, pause/resume, hidden tab, privacy failure, late predictions, verdict switches, queue full, setup, keyboard, mobile, reload');
} finally { await browser.close(); }
