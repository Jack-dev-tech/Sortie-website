const MAX_WIDTH = 640;
const INIT_TIMEOUT = 30000;
const FRAME_TIMEOUT = 10000;

// The raw video is never a rendering or prediction source outside this class.
export class PrivacyCamera {
  constructor(video, canvas, { onFrame, onError }) {
    this.video = video;
    this.canvas = canvas;
    this.onFrame = onFrame;
    this.onError = onError;
    this.generation = 0;
    this.frameId = 0;
    this.available = false;
  }

  start() {
    this.stop();
    const generation = this.generation;
    try {
      if (!globalThis.Worker || !globalThis.OffscreenCanvas || !globalThis.createImageBitmap) {
        throw new Error("This browser cannot run the privacy filter");
      }
      const worker = new Worker(new URL("./privacy-worker.js", import.meta.url));
      this.worker = worker;
      this.lastVideoTime = -1;
      this.pending = null;
      this.timer = setTimeout(() => this.fail("Privacy filter initialization timed out"), INIT_TIMEOUT);
      worker.onerror = (event) => {
        event.preventDefault();
        if (generation === this.generation) this.fail("Privacy worker failed");
      };
      worker.onmessageerror = () => {
        if (generation === this.generation) this.fail("Privacy worker message failed");
      };
      worker.onmessage = ({ data }) => {
        if (generation !== this.generation) {
          data.bitmap?.close();
          return;
        }
        if (data.type === "error") return this.fail(data.message);
        if (data.type === "ready") {
          clearTimeout(this.timer);
          this.schedule(generation);
        } else if (data.type === "frame") {
          try {
            if (data.id !== this.pending || !data.bitmap) {
              throw new Error("Privacy frame mismatch");
            }
            clearTimeout(this.timer);
            this.canvas.width = data.bitmap.width;
            this.canvas.height = data.bitmap.height;
            this.canvas.getContext("2d").drawImage(data.bitmap, 0, 0);
            this.available = true;
            this.pending = null;
            this.onFrame();
            this.schedule(generation);
          } catch (error) {
            this.fail(error.message);
          } finally {
            data.bitmap?.close();
          }
        }
      };
      worker.postMessage({ type: "init" });
    } catch (error) {
      this.fail(error.message);
    }
  }

  schedule(generation) {
    if (generation !== this.generation) return;
    this.raf = requestAnimationFrame(() => this.capture(generation));
  }

  async capture(generation) {
    if (generation !== this.generation || document.hidden) return;
    const { video } = this;
    if (video.readyState < 2 || video.currentTime === this.lastVideoTime) {
      this.schedule(generation);
      return;
    }
    this.lastVideoTime = video.currentTime;
    const width = Math.min(MAX_WIDTH, video.videoWidth);
    const height = Math.max(1, Math.round(video.videoHeight * width / video.videoWidth));
    this.pending = ++this.frameId;
    this.timer = setTimeout(() => this.fail("Privacy frame processing timed out"), FRAME_TIMEOUT);
    try {
      const bitmap = await createImageBitmap(video, { resizeWidth: width, resizeHeight: height });
      if (generation !== this.generation) {
        bitmap.close();
        return;
      }
      try {
        this.worker.postMessage({
          type: "frame", bitmap, timestamp: performance.now(), id: this.pending,
        }, [bitmap]);
      } catch (error) {
        bitmap.close();
        throw error;
      }
    } catch (error) {
      if (generation === this.generation) this.fail(error.message);
    }
  }

  fail(message) {
    this.stop();
    this.onError(Object.assign(new Error(message), { name: "PrivacyError" }));
  }

  stop() {
    this.generation++;
    this.available = false;
    clearTimeout(this.timer);
    cancelAnimationFrame(this.raf);
    this.worker?.terminate();
    this.worker = null;
    this.pending = null;
    this.canvas.getContext("2d")?.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
