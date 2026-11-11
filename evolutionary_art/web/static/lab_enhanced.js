// Genome lab, enhanced part: the function set a mood and style give, and one render traced step by step
// (rotation/scale, three random expressions per layer, blend operations, base colour, style effect, score terms).
import { esc, list } from './core.js';
import { makeRng } from './py_random.js';
import { EnhancedArt, MOODS, STYLES, STYLE_FUNCTIONS, MOOD_FUNCTIONS, BASE_FUNCTIONS, functionSet } from './enhanced_engine.js';
import { drawImage, exportPng, hsvCss, rgbCss, swatch } from './art_view.js';

const EFFECTS = {
  geometric: 'Canny edges (thresholds 100, 200) overlaid: 0.7 × image + 0.3 × edges.',
  organic: 'Gaussian noise (σ 0.1) added, then a bilateral filter (d 9, σ 75).',
  fluid: 'Gaussian blur 15 × 15, then sharpened against a 31 × 31 blur: 1.5 × blur15 − 0.5 × blur31.',
  structured: 'No post-processing effect.',
};

export function mountEnhancedLab(host) {
  host.innerHTML = `
  <h2>Enhanced system: a function set</h2>
  <p>An enhanced individual is just a set of function names: <code>${BASE_FUNCTIONS.join(', ')}</code> plus the ones its
  style and mood add. Each render draws three new random expressions from that set (depth 3 when energetic, else 2),
  shifted and scaled at random, blends them, tints them with a random base colour and applies the style's effect.
  Crossover takes the union of two parents' sets; mutation adds 1 to 3 of <code>sin, cos, wave, flow, turbulence</code>.</p>
  <div class="scroll"><table class="ftable"><tr><th>style</th><th>adds</th></tr>
    ${list(STYLES, s => `<tr><td>${s}</td><td>${STYLE_FUNCTIONS[s].join(', ')}</td></tr>`)}
    <tr><th>mood</th><th>adds</th></tr>${list(MOODS, m => `<tr><td>${m}</td><td>${MOOD_FUNCTIONS[m].join(', ')}</td></tr>`)}</table></div>
  <div class="lab">
    <div class="panel">
      <div class="row2">
        <div><label for="le-mood">Mood</label><select id="le-mood">${list(MOODS, m => `<option>${m}</option>`)}</select></div>
        <div><label for="le-style">Style</label><select id="le-style">${list(STYLES, s => `<option>${s}</option>`)}</select></div>
      </div>
      <label for="le-seed">Seed</label><input id="le-seed" type="number" min="0" value="5">
      <div class="btns"><button type="button" id="le-go">Render</button><button type="button" id="le-next" class="alt">Next seed</button>
        <button type="button" id="le-png" class="alt">Export PNG</button></div>
      <div id="le-set"></div>
    </div>
    <div><canvas id="le-img" class="lab-big" role="img" aria-label="Enhanced render"></canvas><div id="le-trace" class="small"></div></div>
  </div>`;
  const $ = s => host.querySelector(s);
  function go() {
    const mood = $('#le-mood').value, style = $('#le-style').value, seed = +$('#le-seed').value || 0;
    const art = new EnhancedArt(makeRng(seed), { width: 192, height: 192, mood, style });
    const ind = art.newIndividual(), trace = [], parts = {};
    const img = art.createArtwork([ind], trace);
    art.evaluateAesthetics(img, parts);
    drawImage($('#le-img'), img);
    const t = trace[0];
    $('#le-set').innerHTML = `<h3>Function set</h3>${list(functionSet({ mood, style }), f => `<span class="tag a">${esc(f)}</span>`)}
      <p class="small">Colour theme ${list(art.intent.color_theme, c => swatch(hsvCss(c)))}</p>`;
    $('#le-trace').innerHTML = `<ol class="trace">
      ${list(t.exprs, (e, i) => `<li>Layer ${i + 1}: <code>${esc(e)}</code>${i ? ` <span class="tag b">${esc(t.ops[i - 1] || '')}</span>` : ''}</li>`)}
      <li>Tint: base colour ${t.base ? swatch(rgbCss(t.base.map(v => v * 255))) : '(none)'} × variation
        ${t.variation ? t.variation.map(v => v.toFixed(2)).join(', ') : ''}</li>
      <li>${esc(style)}: ${EFFECTS[style]} Then contrast ×1.5 around 0.5.</li>
      <li>Score ${parts.score.toFixed(4)}: complexity ${parts.complexity.toFixed(3)}, composition ${parts.composition.toFixed(3)},
        colour harmony ${parts.color_harmony.toFixed(3)} (× a random 0.95 to 1.05).</li></ol>`;
  }
  $('#le-go').addEventListener('click', go);
  $('#le-next').addEventListener('click', () => { $('#le-seed').value = +$('#le-seed').value + 1; go(); });
  ['#le-mood', '#le-style', '#le-seed'].forEach(s => $(s).addEventListener('change', go));
  $('#le-png').addEventListener('click', () => exportPng($('#le-img'), 'enhanced_render.png'));
  go();
}
