// Canvas helpers shared by the pages: draw an engine image ({ w, h, data } floats in [0, 1]) exactly as the Python
// package saves it ((img * 255).astype(uint8)), PNG export, colour swatches and the fitness-history chart.
import { toUint8 } from './cv_ops.js';
import { hsvToRgb } from './py_random.js';
import { download } from './core.js';

export function drawImage(canvas, img) {
  canvas.width = img.w; canvas.height = img.h;
  const ctx = canvas.getContext('2d'), id = ctx.createImageData(img.w, img.h), u8 = toUint8(img.data, img.f32);
  for (let i = 0, n = img.w * img.h; i < n; i++) {
    id.data[i * 4] = u8[i * 3]; id.data[i * 4 + 1] = u8[i * 3 + 1]; id.data[i * 4 + 2] = u8[i * 3 + 2]; id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return canvas;
}

export function canvasFor(img, cls = '', label = '') {
  const c = document.createElement('canvas');
  if (cls) c.className = cls;
  c.setAttribute('role', 'img');
  if (label) c.setAttribute('aria-label', label);
  return drawImage(c, img);
}

export function exportPng(canvas, name) {
  canvas.toBlob(b => download(name, b, 'image/png'), 'image/png');
}

export const rgbCss = rgb => `rgb(${rgb.map(v => Math.round(v)).join(',')})`;
export const hsvCss = ([h, s, v]) => rgbCss(hsvToRgb(h, s, v).map(c => c * 255));
export const swatch = (css, title = '') => `<span class="sw" style="background:${css}" title="${title}"></span>`;

// Line chart of best and mean score per generation.
export function drawHistory(canvas, hist) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 300, H = canvas.clientHeight || 120;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue('--line').trim(), mute = css.getPropertyValue('--mute').trim();
  ctx.strokeStyle = line; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  if (!hist.length) { ctx.fillStyle = mute; ctx.font = '12px sans-serif'; ctx.fillText('Scores appear after the first generation', 10, H / 2); return; }
  const vals = hist.flatMap(h => [h.best, h.mean]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) { hi += 0.01; lo -= 0.01; }
  const pad = 22, x = i => pad + (hist.length < 2 ? (W - 2 * pad) / 2 : (i / (hist.length - 1)) * (W - 2 * pad));
  const y = v => H - 16 - ((v - lo) / (hi - lo)) * (H - 30);
  const series = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    hist.forEach((h, i) => (i ? ctx.lineTo(x(i), y(h[key])) : ctx.moveTo(x(i), y(h[key]))));
    ctx.stroke();
    hist.forEach((h, i) => { ctx.fillStyle = color; ctx.fillRect(x(i) - 1.5, y(h[key]) - 1.5, 3, 3); });
  };
  series('mean', css.getPropertyValue('--accent3').trim());
  series('best', css.getPropertyValue('--accent').trim());
  ctx.fillStyle = mute; ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(hi.toFixed(3), 4, 12); ctx.fillText(lo.toFixed(3), 4, H - 4);
  ctx.fillText(`gen ${hist[hist.length - 1].generation}`, W - 60, H - 4);
}
