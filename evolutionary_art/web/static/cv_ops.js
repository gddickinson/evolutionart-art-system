// The OpenCV operations the enhanced system uses (rendering.py _apply_style_effects / evaluate_aesthetics), ported
// from OpenCV's algorithms: Canny (3x3 Sobel, L1 magnitude, multi-channel max), GaussianBlur and bilateralFilter
// (float32 path, 4096-bin colour LUT). Border handling as in OpenCV: replicate for Canny, reflect-101 for the filters.

const f32 = Math.fround;

// numpy (img * 255).astype(np.uint8) for values in [0, 1] (truncation). f32: image stored as float32 in Python.
export function toUint8(data, f32math = false) {
  const out = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = f32math ? f32(data[i] * 255) : data[i] * 255;
    out[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
  }
  return out;
}

// Index helpers for borders.
const replicate = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
export function reflect101(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

// cv2.Canny(uint8 image with cn channels, low, high) -> Uint8Array (0 / 255) of h*w.
export function canny(src, w, h, cn, low, high) {
  const n = w * h, dx = new Int32Array(n), dy = new Int32Array(n), mag = new Int32Array(n);
  for (let y = 0; y < h; y++) {
    const ym = replicate(y - 1, h), yp = replicate(y + 1, h);
    for (let x = 0; x < w; x++) {
      const xm = replicate(x - 1, w), xp = replicate(x + 1, w);
      let best = -1, bx = 0, by = 0;
      for (let c = 0; c < cn; c++) {
        const p = (yy, xx) => src[(yy * w + xx) * cn + c];
        const gx = (p(ym, xp) - p(ym, xm)) + 2 * (p(y, xp) - p(y, xm)) + (p(yp, xp) - p(yp, xm));
        const gy = (p(yp, xm) - p(ym, xm)) + 2 * (p(yp, x) - p(ym, x)) + (p(yp, xp) - p(ym, xp));
        const m = Math.abs(gx) + Math.abs(gy);
        if (m > best) { best = m; bx = gx; by = gy; }
      }
      const i = y * w + x; dx[i] = bx; dy[i] = by; mag[i] = best;
    }
  }
  // map: 0 = maybe edge, 1 = not an edge, 2 = edge; 1-pixel border of 1s.
  const mw = w + 2, map = new Uint8Array(mw * (h + 2)).fill(1), stack = [];
  const M = (y, x) => (y < 0 || y >= h || x < 0 || x >= w ? 0 : mag[y * w + x]);
  const SHIFT = 15, TG22 = Math.trunc(0.4142135623730950488016887242097 * (1 << SHIFT) + 0.5);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const m = mag[y * w + x];
      if (m <= low) continue;
      const xs = dx[y * w + x], ys = dy[y * w + x];
      const ax = Math.abs(xs), ay = Math.abs(ys) * (1 << SHIFT), tg22x = ax * TG22;
      let keep;
      if (ay < tg22x) keep = m > M(y, x - 1) && m >= M(y, x + 1);
      else {
        const tg67x = tg22x + ax * (1 << (SHIFT + 1));
        if (ay > tg67x) keep = m > M(y - 1, x) && m >= M(y + 1, x);
        else {
          const s = (xs ^ ys) < 0 ? -1 : 1;
          keep = m > M(y - 1, x - s) && m > M(y + 1, x + s);
        }
      }
      if (!keep) continue;
      const k = (y + 1) * mw + x + 1;
      if (m > high) { map[k] = 2; stack.push(k); } else map[k] = 0;
    }
  }
  const nb = [-mw - 1, -mw, -mw + 1, -1, 1, mw - 1, mw, mw + 1];
  while (stack.length) {
    const k = stack.pop();
    for (const o of nb) if (map[k + o] === 0) { map[k + o] = 2; stack.push(k + o); }
  }
  const out = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = map[(y + 1) * mw + x + 1] === 2 ? 255 : 0;
  return out;
}

// cv2.getGaussianKernel(ksize, 0): sigma from the kernel size.
export function gaussianKernel(ksize) {
  const sigma = ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8, scale2 = -0.5 / (sigma * sigma);
  const k = new Float64Array(ksize); let sum = 0;
  for (let i = 0; i < ksize; i++) { const x = i - (ksize - 1) * 0.5; k[i] = Math.exp(scale2 * x * x); sum += k[i]; }
  for (let i = 0; i < ksize; i++) k[i] *= 1 / sum;
  return k;
}

// cv2.GaussianBlur(img, (ksize, ksize), 0) on a float64 (h, w, 3) image (separable, reflect-101 border).
export function gaussianBlur(data, w, h, ksize) {
  const k = gaussianKernel(ksize), r = (ksize - 1) / 2, tmp = new Float64Array(data.length), out = new Float64Array(data.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let i = 0; i < ksize; i++) s += k[i] * data[(y * w + reflect101(x + i - r, w)) * 3 + c];
    tmp[(y * w + x) * 3 + c] = s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let i = 0; i < ksize; i++) s += k[i] * tmp[(reflect101(y + i - r, h) * w + x) * 3 + c];
    out[(y * w + x) * 3 + c] = s;
  }
  return out;
}

// cv2.bilateralFilter(img.astype(float32), d, sigmaColor, sigmaSpace) for a 3-channel image; returns float32 values.
export function bilateralFilter(data, w, h, d, sigmaColor, sigmaSpace) {
  const src = Float32Array.from(data), cn = 3;
  let lo = Infinity, hi = -Infinity;
  for (const v of src) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (Math.abs(lo - hi) < 1.1920928955078125e-7) return src;
  const gc = -0.5 / (sigmaColor * sigmaColor), gs = -0.5 / (sigmaSpace * sigmaSpace);
  const radius = Math.max(d > 0 ? Math.floor(d / 2) : Math.round(sigmaSpace * 1.5), 1);
  const bins = (1 << 12) * cn, len = f32((hi - lo) * cn), scaleIndex = f32(bins / len);
  const lut = new Float32Array(bins + 2); let last = 1;
  for (let i = 0; i < bins + 2; i++) {
    if (last > 0) { const v = i / scaleIndex; lut[i] = Math.exp(v * v * gc); last = lut[i]; } else lut[i] = 0;
  }
  const offs = [], sw = [];
  for (let i = -radius; i <= radius; i++) for (let j = -radius; j <= radius; j++) {
    const r = Math.sqrt(i * i + j * j);
    if (r > radius || (i === 0 && j === 0)) continue;
    sw.push(f32(Math.exp(r * r * gs))); offs.push([i, j]);
  }
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 3, b0 = src[p], g0 = src[p + 1], r0 = src[p + 2];
    let sb = b0, sg = g0, sr = r0, wsum = 1;
    for (let k = 0; k < offs.length; k++) {
      const q = (reflect101(y + offs[k][0], h) * w + reflect101(x + offs[k][1], w)) * 3;
      const b = src[q], g = src[q + 1], r = src[q + 2];
      let alpha = f32(f32(Math.abs(b - b0) + Math.abs(g - g0) + Math.abs(r - r0)) * scaleIndex);
      const idx = Math.floor(alpha); alpha = f32(alpha - idx);
      const wt = f32(sw[k] * f32(lut[idx] + f32(alpha * f32(lut[idx + 1] - lut[idx]))));
      wsum = f32(wsum + wt); sb = f32(sb + f32(b * wt)); sg = f32(sg + f32(g * wt)); sr = f32(sr + f32(r * wt));
    }
    out[p] = f32(sb / wsum); out[p + 1] = f32(sg / wsum); out[p + 2] = f32(sr / wsum);
  }
  return out;
}
