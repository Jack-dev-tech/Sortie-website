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
      this.statusError = 'Capture paused: cannot reach the local server. Reconnecting…';
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
    this.retry.hidden = !(s?.error || this.localError);
    this.retry.disabled = !s?.configured;
    const link = document.querySelector('[data-roboflow]');
    link.hidden = !s?.project_url;
    if (s?.project_url) link.href = s.project_url;
    for (const key of ['captured', 'uploaded', 'pending']) {
      document.querySelector(`[data-count-${key}]`).textContent = s ? s[key] : '—';
    }
    this.message.textContent = this.statusError || s?.setup || this.localError ||
      (s?.full ? 'Queue full: 500 images waiting. Capture resumes when uploads make room.' : null) ||
      (s?.failed ? s.error : null) ||
      (document.hidden ? 'Capture paused while this tab is hidden.' : null) ||
      (!this.ready ? 'Capture paused until the camera is ready.' : null) ||
      (this.paused ? 'Capture paused. Saved images continue uploading.' : null) ||
      (this.pending ? 'Saving capture to the local queue…' : null) ||
      s?.error || (s ? 'Watching for motion · at most one image every 3 seconds' : 'Checking Roboflow setup…');
  }

  tick(motion, capture, ready) {
    if (this.ready !== ready) { this.ready = ready; this.render(); }
    if (!this.active || !ready || document.hidden || this.paused || !this.status?.can_capture ||
        this.pending || !motion || Date.now() - this.lastCapture < 3000) return;
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
      const img = document.createElement('img');
      img.src = capture.image;
      img.alt = 'Capture saved for manual labeling';
      const recent = document.querySelector('[data-training-recent]');
      document.querySelector('[data-training-empty]').hidden = true;
      recent.prepend(img);
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
