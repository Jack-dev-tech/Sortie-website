// Pure mask operations shared by the worker and regression tests.
export function personMask(confidence, width, height, threshold = 0.25) {
  if (confidence.length !== width * height) throw new Error("Invalid person mask");
  const mask = new Uint8Array(confidence.length);
  for (let i = 0; i < mask.length; i++) {
    if (!Number.isFinite(confidence[i])) throw new Error("Invalid person confidence");
    mask[i] = confidence[i] >= threshold ? 1 : 0;
  }
  return mask;
}

export function addBox(mask, width, height, box, padding = 0.2) {
  if (![box.x, box.y, box.w, box.h].every(Number.isFinite) || box.w < 0 || box.h < 0) {
    throw new Error("Invalid privacy region");
  }
  const clamp = (value, max) => Math.max(0, Math.min(max, value));
  const left = clamp(Math.floor((box.x - box.w * padding) * width), width);
  const right = clamp(Math.ceil((box.x + box.w * (1 + padding)) * width), width);
  const top = clamp(Math.floor((box.y - box.h * padding) * height), height);
  const bottom = clamp(Math.ceil((box.y + box.h * (1 + padding)) * height), height);
  for (let y = top; y < bottom; y++) mask.fill(1, y * width + left, y * width + right);
}

// A summed-area table grows the mask without leaving gaps at body boundaries.
export function dilateMask(mask, width, height, radius = 3) {
  const stride = width + 1;
  const sums = new Uint32Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    let row = 0;
    for (let x = 0; x < width; x++) {
      row += mask[y * width + x];
      sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + row;
    }
  }
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(width, x + radius + 1);
      const y0 = Math.max(0, y - radius), y1 = Math.min(height, y + radius + 1);
      const sum = sums[y1 * stride + x1] - sums[y0 * stride + x1]
        - sums[y1 * stride + x0] + sums[y0 * stride + x0];
      if (sum) pixels[(y * width + x) * 4 + 3] = 255;
    }
  }
  return pixels;
}

// Hand exceptions are applied after dilation. Faces always take priority,
// including the padding around them, even when the person segmenter misses one.
export function privacyMask(body, width, height, faces = [], hands = []) {
  const pixels = dilateMask(body, width, height);
  const faceMask = new Uint8Array(width * height);
  const handMask = new Uint8Array(width * height);
  for (const box of faces) addBox(faceMask, width, height, box, 0.25);
  for (const box of hands) addBox(handMask, width, height, box, 0.3);
  const protectedFaces = dilateMask(faceMask, width, height);
  for (let i = 0; i < handMask.length; i++) {
    const alpha = i * 4 + 3;
    if (protectedFaces[alpha]) pixels[alpha] = 255;
    else if (handMask[i]) pixels[alpha] = 0;
  }
  return pixels;
}
