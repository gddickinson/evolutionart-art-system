// Evolve page (#/evolve): run either system generation by generation, see the whole scored population, follow the
// best and mean score, pick any individual to inspect, export PNGs. Query: ?mode=basic|enhanced&seed=&size=&mood=&style=
import { route, app, esc, list, onLeave } from './core.js';
import { MOODS, STYLES } from './enhanced_engine.js';
import { ArtRun } from './evolve_run.js';
import { drawImage, canvasFor, exportPng, drawHistory, hsvCss, rgbCss, swatch } from './art_view.js';

const SIZES = [64, 96, 128, 192, 256];
const opt = (v, cur, label = v) => `<option value="${esc(v)}"${String(v) === String(cur) ? ' selected' : ''}>${esc(label)}</option>`;

route('evolve', (args, q) => {
  const cfg = { mode: q.mode === 'basic' ? 'basic' : 'enhanced', seed: +q.seed || 7, size: SIZES.includes(+q.size) ? +q.size : 96,
    mood: MOODS.includes(q.mood) ? q.mood : '', style: STYLES.includes(q.style) ? q.style : '', stopAt: 50 };
  app().innerHTML = `
  <h1 class="glow">Evolve</h1>
  <p class="lead">Each generation scores a population of 20 images with the system's fitness function and breeds the
  next one. Click any image to inspect its genome; the large view follows the best of each generation.</p>
  <div class="evo">
    <aside class="panel evo-ctl">
      <label>System</label>
      <div class="seg" role="group" aria-label="System">
        <button type="button" data-mode="enhanced">Enhanced</button><button type="button" data-mode="basic">Basic</button>
      </div>
      <div class="enh-only">
        <label for="ev-mood">Mood</label><select id="ev-mood">${opt('', cfg.mood, 'random')}${list(MOODS, m => opt(m, cfg.mood))}</select>
        <label for="ev-style">Style</label><select id="ev-style">${opt('', cfg.style, 'random')}${list(STYLES, s => opt(s, cfg.style))}</select>
      </div>
      <div class="row2">
        <div><label for="ev-seed">Seed</label><input id="ev-seed" type="number" min="0" step="1" value="${cfg.seed}"></div>
        <div><label for="ev-size">Size (px)</label><select id="ev-size">${list(SIZES, s => opt(s, cfg.size))}</select></div>
      </div>
      <label for="ev-stop">Stop after generation</label><input id="ev-stop" type="number" min="1" max="500" value="${cfg.stopAt}">
      <div class="btns">
        <button type="button" id="ev-new">New run</button><button type="button" id="ev-dice" class="alt" aria-label="New run with a random seed">Random seed</button>
        <button type="button" id="ev-step" class="alt">Step</button><button type="button" id="ev-play">Run</button>
      </div>
      <div id="ev-aim" class="aim small"></div>
    </aside>
    <section class="evo-main">
      <div class="stage-row">
        <figure class="big"><canvas id="ev-big" role="img" aria-label="Selected artwork"></canvas>
          <figcaption id="ev-cap" class="small mute">Press Step or Run to evolve the first generation.</figcaption></figure>
        <div class="evo-side">
          <div class="stats" id="ev-stats"></div>
          <canvas id="ev-hist" class="hist" role="img" aria-label="Best and mean score per generation"></canvas>
          <p class="small mute"><span class="key a"></span>best <span class="key c"></span>mean score per generation</p>
          <button type="button" id="ev-png" class="alt">Export PNG</button>
          <div id="ev-genome" class="genome"></div>
        </div>
      </div>
      <h2>Population</h2>
      <div id="ev-pop" class="pop"></div>
    </section>
  </div>`;

  const $ = s => app().querySelector(s);
  let run = null, timer = 0, playing = false, sel = 0;

  const setMode = m => {
    cfg.mode = m;
    app().querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
    $('.enh-only').hidden = m !== 'enhanced';
  };
  const stop = () => { playing = false; clearTimeout(timer); $('#ev-play').textContent = 'Run'; };
  onLeave(() => { playing = false; clearTimeout(timer); });

  function fresh() {
    stop();
    cfg.seed = Math.max(0, Math.trunc(+$('#ev-seed').value || 0));
    run = new ArtRun({ mode: cfg.mode, seed: cfg.seed, size: +$('#ev-size').value, mood: $('#ev-mood').value, style: $('#ev-style').value });
    sel = 0;
    const d = run.describe();
    $('#ev-aim').innerHTML = d.intent
      ? `<p><b>Intent</b> ${esc(d.intent.mood)} · ${esc(d.intent.style)}<br>Colour theme ${list(d.intent.color_theme, c => swatch(hsvCss(c)))}</p>`
      : `<p><b>Target palettes</b> (colour harmony)</p>${list(d.palettes, p => `<div>${list(p, c => swatch(rgbCss(c)))}</div>`)}`;
    $('#ev-pop').innerHTML = ''; $('#ev-genome').innerHTML = ''; $('#ev-stats').innerHTML = '';
    $('#ev-cap').textContent = 'Press Step or Run to evolve the first generation.';
    const big = $('#ev-big'); big.width = big.height = run.size; big.getContext('2d').clearRect(0, 0, run.size, run.size);
    drawHistory($('#ev-hist'), []);
  }

  function show(i) {
    const res = run.last; if (!res) return;
    sel = i;
    const it = res.items[i];
    drawImage($('#ev-big'), it.img);
    $('#ev-cap').textContent = `Generation ${res.generation}, individual ${i + 1} of ${res.items.length}${i === res.bestIdx ? ' (best)' : ''}: score ${it.score.toFixed(4)}`;
    const parts = it.parts ? Object.entries(it.parts).filter(([k]) => k !== 'fitness' && k !== 'score')
      .map(([k, v]) => `<span class="tag">${esc(k.replace('_', ' '))} ${v.toFixed(3)}</span>`).join('') : '';
    $('#ev-genome').innerHTML = run.mode === 'basic'
      ? `<h3>Genome (R, G, B expressions)</h3>${list(it.genome, (g, c) => `<code class="expr"><b>${'RGB'[c]}</b> ${esc(g)}</code>`)}<div>${parts}</div>`
      : `<h3>Genome (function set)</h3><div>${list(it.genome, f => `<span class="tag a">${esc(f)}</span>`)}</div><div>${parts}</div>`;
    app().querySelectorAll('.pop button').forEach((b, k) => b.classList.toggle('sel', k === i));
  }

  function render(res) {
    const pop = $('#ev-pop'); pop.innerHTML = '';
    res.items.forEach((it, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'cell' + (i === res.bestIdx ? ' best' : '');
      b.setAttribute('aria-label', `Individual ${i + 1}, score ${it.score.toFixed(3)}`);
      b.appendChild(canvasFor(it.img));
      b.insertAdjacentHTML('beforeend', `<span>${it.score.toFixed(3)}</span>`);
      b.addEventListener('click', () => show(i));
      pop.appendChild(b);
    });
    const h = run.history[run.history.length - 1];
    $('#ev-stats').innerHTML = `<div class="tiles mini">
      <div class="tile"><b>${res.generation}</b><span>generation</span></div>
      <div class="tile"><b>${h.best.toFixed(3)}</b><span>best score</span></div>
      <div class="tile"><b>${res.mutationRate == null ? '0.2' : res.mutationRate.toFixed(1)}</b><span>${res.mutationRate == null ? 'mutation prob.' : 'mutation rate'}</span></div></div>`;
    drawHistory($('#ev-hist'), run.history);
    show(res.bestIdx);
  }

  function stepOnce() { render(run.step()); }
  function tick() {
    if (!playing) return;
    if (run.generation + 1 >= +$('#ev-stop').value) { stop(); return; }
    stepOnce();
    timer = setTimeout(tick, 30);
  }

  app().querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => { setMode(b.dataset.mode); fresh(); }));
  $('#ev-new').addEventListener('click', fresh);
  $('#ev-dice').addEventListener('click', () => { $('#ev-seed').value = Math.floor(Math.random() * 1e6); fresh(); });
  $('#ev-step').addEventListener('click', () => { stop(); stepOnce(); });
  $('#ev-play').addEventListener('click', () => {
    if (playing) return stop();
    playing = true; $('#ev-play').textContent = 'Pause'; tick();
  });
  $('#ev-png').addEventListener('click', () => {
    if (!run.last) return;
    exportPng($('#ev-big'), `${run.mode}_seed${run.seed}_gen${run.generation}_ind${sel + 1}.png`);
  });
  ['#ev-mood', '#ev-style', '#ev-size'].forEach(s => $(s).addEventListener('change', fresh));
  setMode(cfg.mode);
  fresh();
  if (q.run) $('#ev-play').click(); else stepOnce();
});
