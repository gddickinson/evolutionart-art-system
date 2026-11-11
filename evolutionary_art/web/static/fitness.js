// Port of evolutionary_art/fitness.py plus the numpy reductions the fitness code uses (mean, std, np.gradient).
// Images are { w, h, data } with data a Float64Array of h*w*3 values, row-major [y][x][channel] like numpy (H, W, 3).

export const makeImage = (w, h) => ({ w, h, data: new Float64Array(w * h * 3) });

export function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }

export function std(a) {
  const m = mean(a); let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; }
  return Math.sqrt(s / a.length);
}

// Copy of img[y0:y1, x0:x1] (all channels), flattened.
export function region(img, y0, y1, x0, x1) {
  const out = []; const { w, data } = img;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) for (let c = 0; c < 3; c++) out.push(data[(y * w + x) * 3 + c]);
  return Float64Array.from(out);
}

// fitness.evaluate_color_harmony: best palette's mean of exp(-10 * distance from each palette colour to its nearest pixel).
export function evaluateColorHarmony(img, palettes) {
  const d = img.data, n = d.length / 3;
  let maxHarmony = 0;
  for (const palette of palettes) {
    let harmony = 0;
    for (const col of palette) {
      const r = col[0] / 255.0, g = col[1] / 255.0, b = col[2] / 255.0;
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        const dr = d[i * 3] - r, dg = d[i * 3 + 1] - g, db = d[i * 3 + 2] - b;
        const dist = Math.sqrt(dr * dr + dg * dg + db * db);
        if (dist < best) best = dist;
      }
      harmony += Math.exp(-best * 10);
    }
    maxHarmony = Math.max(maxHarmony, harmony / palette.length);
  }
  return maxHarmony;
}

export const evaluateComplexity = img => std(img.data);

// fitness.evaluate_balance (also inlined in EvolutionaryArtSystem.fitness_components).
export function evaluateBalance(img) {
  const half = Math.floor(img.w / 2);
  return 1 - Math.abs(mean(region(img, 0, img.h, 0, half)) - mean(region(img, 0, img.h, half, img.w)));
}

// fitness.evaluate_composition_thirds: mean std of 20x20 windows around the four rule-of-thirds points.
export function evaluateCompositionThirds(img) {
  const { w, h } = img, th = Math.floor(h / 3), tw = Math.floor(w / 3);
  const pts = [[tw, th], [2 * tw, th], [tw, 2 * th], [2 * tw, 2 * th]];
  let score = 0;
  for (const [px, py] of pts) {
    score += std(region(img, Math.max(0, py - 10), Math.min(h, py + 10), Math.max(0, px - 10), Math.min(w, px + 10)));
  }
  return score / pts.length;
}

// np.gradient along one axis of an array with the given shape (edge_order=1), returned as |gradient| summed.
function absGradSum(a, shape, axis) {
  const strides = []; let st = 1;
  for (let i = shape.length - 1; i >= 0; i--) { strides[i] = st; st *= shape[i]; }
  const n = shape[axis], s = strides[axis];
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const k = Math.floor(i / s) % n;
    let g;
    if (n < 2) g = 0;
    else if (k === 0) g = a[i + s] - a[i];
    else if (k === n - 1) g = a[i] - a[i - s];
    else g = (a[i + s] - a[i - s]) / 2;
    sum += Math.abs(g);
  }
  return sum;
}

// np.mean(np.abs(np.gradient(a))) for an n-d array: the mean over every axis's gradient.
export function meanAbsGradient(a, shape) {
  let s = 0;
  for (let ax = 0; ax < shape.length; ax++) s += absGradSum(a, shape, ax);
  return s / (a.length * shape.length);
}
