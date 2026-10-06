// Captures webcam frames for manual labeling. Never runs inference.
export class TrainingCapture {
  constructor() {
    this.active = false;
    this.paused = false;
    this.ready = false;
    this.status = null;
    this.session = null;
    this.lastCapture = -Infinity;
    this.pending = null;
    this.sending = false;
    this.retryAt = 0;
    this.panel = document.querySelector('[data-training]');
    this.message = document.querySelector('[data-training-status]');
    this.pause = document.querySelector('[data-training-pause]');
    this.retry = document.querySelector('[data-training-retry]');
    this.pause.addEventListener('click', () => {
      this.paused = !this.paused;
      this.render();
    });
    this.retry.addEventListener('click', async () => {
      this.retry.disabled = true;
      try {
        const res = await fetch('/training/retry', { method: 'POST', signal: AbortSignal.timeout(10000) });
        if (!res.ok) throw new Error();
        this.retryAt = 0;
        this.localError = null;
      } catch { this.localError = 'Could not retry uploads. Check the local server connection.'; }
      await this.refresh();
    });
    setInterval(() => {
      if (document.hidden) return;
      if (this.active) this.refresh();
      // Finish saving an already captured frame even after switching to Normal.
      if (this.pending && Date.now() >= this.retryAt) this.submit();
    }, 2000);
  }

  select(active) {
    this.active = active;
    this.panel.hidden = !active;
    if (active) {
      this.session = crypto.randomUUID();
      this.paused = false;
      this.status = null;
      this.refresh();
    }
    this.render();
  }

  async refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const res = await fetch('/training/status', { cache: 'no-store', signal: AbortSignal.timeout(10000) });
      if (!res.ok) throw new Error();
      this.status = await res.json();
      this.statusError = null;
    } catch {
      this.status = null;
      this.statusError = 'Reconnecting automatically. Capture is paused until it answers.';
    } finally {
      this.refreshing = false;
      this.render();
    }
  }

  render() {
    if (!this.active) return;
    const s = this.status;
    this.pause.textContent = this.paused ? 'Resume capture' : 'Pause capture';
    this.pause.setAttribute('aria-pressed', String(this.paused));
    this.pause.disabled = !s?.configured;
    const link = document.querySelector('[data-label-studio]');
    link.hidden = !s?.project_url;
    if (s?.project_url) link.href = s.project_url;
    for (const key of ['captured', 'uploaded', 'pending', 'failed']) {
      document.querySelector(`[data-count-${key}]`).textContent = s ? s[key] : '—';
    }
    document.querySelector('[data-count-capacity]').textContent = s?.capacity ? ` / ${s.capacity}` : '';
    document.querySelector('[data-count-failed-cell]').hidden = !(s?.failed > 0);

    const { tone, title, detail, action } = this.describe();
    this.message.dataset.tone = tone;
    this.message.querySelector('[data-status-title]').textContent = title;
    const detailEl = this.message.querySelector('[data-status-detail]');
    detailEl.textContent = detail || '';
    detailEl.hidden = !detail;
    this.retry.hidden = action !== 'retry';
    this.retry.disabled = !s?.configured;
  }

  // The one thing the operator most needs to know right now, most urgent first.
  // tone: stop = capture can't happen until something is fixed, wait = it will
  // resume by itself, ok = capturing normally.
  describe() {
    const s = this.status;
    if (this.statusError) return { tone: 'stop', title: "Can't reach the Sortie server", detail: this.statusError };
    if (s?.setup) return { tone: 'stop', title: "Label Studio isn't connected", detail: s.setup };
    if (this.localError) return { tone: 'stop', title: 'Capture not saved', detail: this.localError, action: 'retry' };
    if (s?.full) return { tone: 'wait', title: 'Queue full',
      detail: `${s.capacity} images waiting. Capture resumes when uploads make room.` };
    if (s?.failed) return { tone: 'stop', title: 'Uploads stopped', detail: s.error, action: 'retry' };
    if (document.hidden) return { tone: 'wait', title: 'Paused while this tab is hidden' };
    if (!this.ready) return { tone: 'wait', title: 'Waiting for the camera' };
    if (this.paused) return { tone: 'wait', title: 'Capture paused', detail: 'Saved images keep uploading.' };
    if (this.pending) return { tone: 'ok', title: 'Saving capture…' };
    if (s?.error) return { tone: 'wait', title: 'Upload retrying', detail: s.error, action: 'retry' };
    if (s) return { tone: 'ok', title: 'Watching for a hand moving an item',
      detail: 'At most one image every 3 seconds.' };
    return { tone: 'wait', title: 'Checking Label Studio…' };
  }

  // `trigger` is true while a hand is moving an item in view; app.js decides that.
  tick(trigger, capture, ready) {
    if (this.ready !== ready) { this.ready = ready; this.render(); }
    if (!this.active || !ready || document.hidden || this.paused || !this.status?.can_capture ||
        this.pending || !trigger || Date.now() - this.lastCapture < 3000) return;
    const image = capture();
    if (!image) return;
    this.lastCapture = Date.now();
    this.pending = { id: crypto.randomUUID(), session: this.session, image };
    this.submit();
  }

  async submit() {
    if (this.sending || !this.pending) return;
    this.sending = true;
    const capture = this.pending;
    try {
      const res = await fetch('/training/captures', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(capture), signal: AbortSignal.timeout(15000),
      });
      const data = await res.json();
      if (!res.ok) {
        this.localError = data.error || 'Could not save capture.';
        if ([400, 409, 413].includes(res.status)) {
          this.pending = null;
          this.paused = true;
        }
        throw new Error();
      }
      this.pending = null;
      this.localError = null;
      const item = document.createElement('li');
      const img = document.createElement('img');
      const time = document.createElement('time');
      const at = new Date(this.lastCapture);
      img.src = capture.image;
      img.alt = 'Capture saved for manual labeling';
      time.dateTime = at.toISOString();
      time.textContent = at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
      item.append(img, time);
      const recent = document.querySelector('[data-training-recent]');
      document.querySelector('[data-training-empty]').hidden = true;
      recent.prepend(item);
      while (recent.children.length > 6) recent.lastChild.remove();
    } catch {
      this.localError ||= 'Capture waiting to be saved. Check the local server; retrying automatically. Keep this page open.';
      this.retryAt = Date.now() + 5000;
    } finally {
      this.sending = false;
      this.refresh();
      this.render();
    }
  }
}
