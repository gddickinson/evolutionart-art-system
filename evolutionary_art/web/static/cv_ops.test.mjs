// node cv_ops.test.mjs: OpenCV ports (Canny, GaussianBlur, bilateralFilter) and fitness helpers vs cv2 / numpy output.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { canny, toUint8, gaussianBlur, bilateralFilter } from './cv_ops.js';
import { evaluateCompositionThirds, meanAbsGradient } from './fitness.js';

const F = JSON.parse(readFileSync(new URL('../../../tests/fixtures/evo_reference.json', import.meta.url))).cv;
const [w, h] = F.size, img = Float64Array.from(F.image);
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

const u8 = toUint8(img);
for (const [key, lo, hi] of [['canny', 100, 200], ['canny_low', 20, 60]]) {
  const e = canny(u8, w, h, 3, lo, hi), on = F[key].filter(v => v).length;
  assert.deepEqual([...e], F[key], `${key}: edge maps differ`);
  console.log(`  ${key}: identical (${on} edge pixels)`);
}
{
  const e = canny(toUint8(Float64Array.from(F.smooth)), 48, 36, 3, 30, 60), on = F.canny_smooth.filter(v => v).length;
  assert.deepEqual([...e], F.canny_smooth, 'canny_smooth: edge maps differ');
  console.log(`  canny_smooth: identical (${on} edge pixels of ${48 * 36})`);
}
const b15 = gaussianBlur(img, w, h, 15), b31 = gaussianBlur(img, w, h, 31);
const d15 = maxDiff(b15, F.blur15), d31 = maxDiff(b31, F.blur31);
assert.ok(d15 < 1e-12 && d31 < 1e-12, `GaussianBlur differs: ${d15} ${d31}`);
const bl = bilateralFilter(img, w, h, 9, 75, 75), dbl = maxDiff(bl, F.bilateral);
assert.ok(dbl < 1e-5, `bilateralFilter differs by ${dbl}`);
const th = evaluateCompositionThirds({ w, h, data: img });
assert.ok(Math.abs(th - F.thirds) < 1e-12, 'composition thirds');
assert.ok(Math.abs(meanAbsGradient(img, [h, w, 3]) - F.gradient_mean) < 1e-12, 'np.gradient mean');
console.log(`cv_ops: ok (Canny identical, GaussianBlur max diff ${Math.max(d15, d31).toExponential(1)}, bilateral (float32) ${dbl.toExponential(1)})`);
