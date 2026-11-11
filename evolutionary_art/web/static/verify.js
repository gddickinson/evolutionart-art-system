// Python vs JavaScript (#/verify): replay the seeded reference cases of the Python package (api/reference) in the
// browser port and measure the differences; on the local app, also run the Python package live next to the JS.
import { route, app, api, esc, list, isStatic, needsServer } from './core.js';
import { PyRandom, NpRandom } from './py_random.js';
import { generateFunction, mutateFunction, renderIndividual, BasicSystem } from './basic_engine.js';
import { EnhancedArt, MOODS, STYLES } from './enhanced_engine.js';
import { canny, toUint8, gaussianBlur, bilateralFilter } from './cv_ops.js';
import { ArtRun } from './evolve_run.js';
import { canvasFor } from './art_view.js';

const rng = s => ({ py: new PyRandom(s), np: new NpRandom(s) });
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const imgOf = (w, h, px) => ({ w, h, data: Float64Array.from(px) });

function checks(R) {
  const rows = [], B = R.basic, E = R.enhanced, C = R.cv, [w, h] = B.size;
  const row = (what, ok, detail) => rows.push({ what, ok, detail });
  const fn = B.functions.filter((s, i) => generateFunction(new PyRandom(i)) === s).length;
  row('Basic: random genomes (expression strings)', fn === B.functions.length, `${fn} / ${B.functions.length} identical`);
  const mu = B.mutations.filter(m => mutateFunction(new PyRandom(m.seed), m.before) === m.after).length;
  row('Basic: mutation', mu === B.mutations.length, `${mu} / ${B.mutations.length} identical`);
  let px = 0, fit = 0;
  for (const ind of B.system.individuals) {
    px = Math.max(px, maxDiff(renderIndividual(ind.funcs, w, h).data, ind.pixels));
    const s = new BasicSystem(rng(B.system.seed), { width: w, height: h });
    fit = Math.max(fit, Math.abs(s.fitnessComponents(renderIndividual(ind.funcs, w, h)).fitness - ind.parts.fitness));
  }
  row('Basic: rendered pixels and fitness', px < 1e-9 && fit < 1e-9, `max pixel diff ${px.toExponential(1)}, fitness ${fit.toExponential(1)}`);
  const run = new BasicSystem(rng(B.run.seed), { width: w, height: h }); run.initialize();
  let same = 0;
  for (const g of B.run.generations) { const s = run.step(); if (JSON.stringify(s.best.funcs) === JSON.stringify(g.best)) same++; }
  row(`Basic: ${B.run.generations.length}-generation run (seed ${B.run.seed})`, same === B.run.generations.length, `best genome identical in ${same} / ${B.run.generations.length} generations`);
  const [cw, ch] = C.size, cimg = Float64Array.from(C.image);
  const cOk = JSON.stringify([...canny(toUint8(cimg), cw, ch, 3, 100, 200)]) === JSON.stringify(C.canny);
  row('OpenCV Canny edges', cOk, cOk ? 'identical edge map' : 'edge maps differ');
  const gb = maxDiff(gaussianBlur(cimg, cw, ch, 15), C.blur15), bl = maxDiff(bilateralFilter(cimg, cw, ch, 9, 75, 75), C.bilateral);
  row('OpenCV GaussianBlur / bilateralFilter', gb < 1e-12 && bl < 1e-5, `max diff ${gb.toExponential(1)} / ${bl.toExponential(1)} (float32)`);
  const shots = [];
  for (const a of E.artworks) {
    const art = new EnhancedArt(rng(a.seed), { width: E.size[0], height: E.size[1], style: a.style });
    const img = art.createArtwork([art.newIndividual()]), d = maxDiff(img.data, a.pixels);
    const sc = Math.abs(art.evaluateAesthetics(img) - a.score);
    row(`Enhanced: ${a.style} render and score (seed ${a.seed})`, d < 1e-5 && sc < 1e-5, `max pixel diff ${d.toExponential(1)}, score diff ${sc.toExponential(1)}`);
    shots.push({ label: `${a.style} · seed ${a.seed}`, py: imgOf(E.size[0], E.size[1], a.pixels), js: img });
  }
  for (const r of E.runs) {
    const art = new EnhancedArt(rng(r.seed), { width: E.size[0], height: E.size[1], style: r.style });
    let ok = 0;
    for (const g of r.generations) { const s = art.step(); if (s.bestIdx === g.best_idx && maxDiff(s.scores, g.scores) < 1e-5) ok++; }
    row(`Enhanced: ${r.style} 3-generation run (seed ${r.seed})`, ok === r.generations.length, `scores and best individual agree in ${ok} / ${r.generations.length} generations`);
  }
  return { rows, shots };
}

route('verify', async () => {
  app().innerHTML = '<h1 class="glow">Python vs JavaScript</h1><p class="mute">Replaying the reference cases...</p>';
  const R = await api('/reference');
  const { rows, shots } = checks(R);
  const pass = rows.filter(r => r.ok).length;
  app().innerHTML = `
  <h1 class="glow">Python vs JavaScript</h1>
  <p class="lead">The browser runs a port of the Python package: Python's Mersenne Twister and numpy's legacy random
  generator, the expression genomes, the fitness functions and the OpenCV filters. The reference cases below were
  produced by the Python package itself (<code>evolutionary_art/web/reference.py</code>) and are replayed here with the
  same seeds. The same comparison runs as node tests in CI.</p>
  <div class="tiles"><div class="tile"><b>${pass} / ${rows.length}</b><span>checks agree</span></div></div>
  <div class="scroll"><table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>
  ${list(rows, r => `<tr><td>${esc(r.what)}</td><td class="${r.ok ? 'ok' : 'err'}">${r.ok ? 'agrees' : 'differs'}</td><td class="mono">${esc(r.detail)}</td></tr>`)}</table></div>
  <p class="small mute">Differences of order 1e-16 are floating-point rounding (V8 vs the C maths library). The organic
  style's bilateral filter works in 32-bit floats in OpenCV, so it agrees to about 1e-7; when such a difference moves
  a pixel across an 8-bit level used by the Canny edge term, an organic score can differ by about 1e-4.</p>
  <h2>Renders side by side (Python left, JavaScript right)</h2>
  <div class="pairs" id="vf-shots"></div>
  <h2>Run the Python package live</h2>
  ${isStatic() ? needsServer('Running the Python package') : `
  <div class="row">
    <div><label for="vf-mode">System</label><select id="vf-mode"><option>enhanced</option><option>basic</option></select></div>
    <div><label for="vf-style">Style</label><select id="vf-style"><option value="">random</option>${list(STYLES, s => `<option>${s}</option>`)}</select></div>
    <div><label for="vf-mood">Mood</label><select id="vf-mood"><option value="">random</option>${list(MOODS, s => `<option>${s}</option>`)}</select></div>
    <div><label for="vf-seed">Seed</label><input id="vf-seed" type="number" min="0" value="3"></div>
  </div>
  <button type="button" id="vf-run">Run 4 generations at 64 px in Python and JavaScript</button>
  <div id="vf-live"></div>`}`;
  const host = document.getElementById('vf-shots');
  for (const s of shots) {
    const fig = document.createElement('figure');
    fig.append(canvasFor(s.py, '', `Python render, ${s.label}`), canvasFor(s.js, '', `JavaScript render, ${s.label}`));
    fig.insertAdjacentHTML('beforeend', `<figcaption>${esc(s.label)}</figcaption>`);
    host.appendChild(fig);
  }
  const btn = document.getElementById('vf-run');
  if (btn) btn.addEventListener('click', () => live(btn));
});

async function live(btn) {
  const v = id => document.getElementById(id).value;
  const req = { mode: v('vf-mode'), seed: +v('vf-seed') || 0, size: 64, generations: 4, mood: v('vf-mood') || null, style: v('vf-style') || null };
  const out = document.getElementById('vf-live');
  btn.disabled = true; out.innerHTML = '<p class="mute">Running the Python package...</p>';
  try {
    const py = await api('/evolve', { body: req });
    const run = new ArtRun(req), rows = [];
    for (const g of py.generations) {
      const s = run.step(), js = s.items[s.bestIdx];
      const img = new Image(); img.src = g.png; img.alt = `Python best of generation ${g.generation}`;
      rows.push({ g, js, img, d: maxDiff(s.items.map(i => i.score), g.scores) });
    }
    out.innerHTML = '<div class="pairs" id="vf-lpairs"></div>';
    const host = document.getElementById('vf-lpairs');
    for (const r of rows) {
      const fig = document.createElement('figure');
      fig.append(r.img, canvasFor(r.js.img, '', `JavaScript best of generation ${r.g.generation}`));
      fig.insertAdjacentHTML('beforeend', `<figcaption>gen ${r.g.generation}: best ${r.g.best_score.toFixed(4)} (Python) vs
        ${r.js.score.toFixed(4)} (JS); max score diff ${r.d.toExponential(1)}</figcaption>`);
      host.appendChild(fig);
    }
  } catch (e) { out.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  btn.disabled = false;
}
