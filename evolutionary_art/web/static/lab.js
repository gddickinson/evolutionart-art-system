// Genome lab (#/lab): the basic genome (three expressions, one per colour channel) with edit / mutate / crossover,
// and the anatomy of one enhanced render (lab_enhanced.js).
import { route, app, esc, list } from './core.js';
import { PyRandom, makeRng } from './py_random.js';
import { generateFunction, mutateFunction, renderIndividual, BasicSystem } from './basic_engine.js';
import { makeImage } from './fitness.js';
import { drawImage, exportPng, swatch, rgbCss } from './art_view.js';
import { mountEnhancedLab } from './lab_enhanced.js';

const SIZE = 192;

function channelImage(img, c) {
  const out = makeImage(img.w, img.h);
  for (let i = 0; i < img.w * img.h; i++) out.data[i * 3 + c] = img.data[i * 3 + c];
  return out;
}

route('lab', () => {
  app().innerHTML = `
  <h1 class="glow">Genome lab</h1>
  <p class="lead">A look inside the two genomes. Everything here calls the same code the Evolve page runs.</p>
  <h2>Basic system: three expressions</h2>
  <p>A basic individual is three <code>ArtisticFunction</code> expressions in <code>x</code> and <code>y</code> (each from
  -1 to 1 across the image), built from <code>+</code>, <code>*</code>, <code>sin</code> and <code>cos</code> to a depth
  of 2 to 4. Each is normalised to 0..1 and becomes the red, green or blue channel. Edit them, or let the operators
  change them: <b>mutate</b> redraws an expression (30%) or scales some of its coefficients by 0.5 to 1.5;
  <b>crossover</b> swaps each channel with a partner's with probability 1/2.</p>
  <div class="lab">
    <div class="panel">
      ${list(['R', 'G', 'B'], (c, i) => `<label for="lab-e${i}">${c} expression</label><input id="lab-e${i}" class="mono" spellcheck="false">`)}
      <p id="lab-err" class="err small" role="alert"></p>
      <div class="row2">
        <div><label for="lab-seed">Seed</label><input id="lab-seed" type="number" min="0" value="3"></div>
        <div><label for="lab-ops">Operator seed</label><input id="lab-ops" type="number" min="0" value="1"></div>
      </div>
      <div class="btns">
        <button type="button" id="lab-new">New genome</button><button type="button" id="lab-mut" class="alt">Mutate</button>
        <button type="button" id="lab-x" class="alt">Crossover</button><button type="button" id="lab-png" class="alt">Export PNG</button>
      </div>
      <div id="lab-x-out" class="small"></div>
    </div>
    <div>
      <canvas id="lab-img" class="lab-big" role="img" aria-label="Rendered genome"></canvas>
      <div class="chans">${list(['red', 'green', 'blue'], (c, i) => `<figure><canvas id="lab-c${i}" role="img" aria-label="${c} channel"></canvas><figcaption>${c}</figcaption></figure>`)}</div>
      <div id="lab-fit" class="small"></div>
    </div>
  </div>
  <div id="lab-enh"></div>`;

  const $ = s => app().querySelector(s);
  const inputs = [0, 1, 2].map(i => $(`#lab-e${i}`));
  const scorer = new BasicSystem(makeRng(0), { width: SIZE, height: SIZE });
  let ops = new PyRandom(1);

  const genome = () => inputs.map(i => i.value.trim());
  function draw() {
    try {
      const img = renderIndividual(genome(), SIZE, SIZE);
      drawImage($('#lab-img'), img);
      [0, 1, 2].forEach(c => drawImage($(`#lab-c${c}`), channelImage(img, c)));
      const p = scorer.fitnessComponents(img);
      $('#lab-fit').innerHTML = `<p>Fitness ${p.fitness.toFixed(4)} = 0.3 × complexity ${p.complexity.toFixed(3)} + 0.3 × balance
        ${p.balance.toFixed(3)} + 0.4 × colour harmony ${p.color_harmony.toFixed(3)}, scored against the palettes of an
        <code>EvolutionaryArtSystem(seed=0)</code>: ${list(scorer.palettes, pal => list(pal, c => swatch(rgbCss(c))), ' · ')}</p>`;
      $('#lab-err').textContent = '';
    } catch (e) { $('#lab-err').textContent = `Cannot evaluate: ${e.message}`; }
  }
  const set = g => { g.forEach((s, i) => { inputs[i].value = s; }); draw(); };
  const newGenome = () => { const py = new PyRandom(+$('#lab-seed').value || 0); set([0, 1, 2].map(() => generateFunction(py))); };
  $('#lab-ops').addEventListener('change', () => { ops = new PyRandom(+$('#lab-ops').value || 0); });
  $('#lab-new').addEventListener('click', () => { $('#lab-seed').value = +$('#lab-seed').value + 1; newGenome(); });
  $('#lab-seed').addEventListener('change', newGenome);
  inputs.forEach(i => i.addEventListener('change', draw));
  $('#lab-mut').addEventListener('click', () => set(genome().map(g => mutateFunction(ops, g))));
  $('#lab-x').addEventListener('click', () => {
    const partner = [0, 1, 2].map(() => generateFunction(ops)), a = genome(), b = [...partner];
    const swapped = [0, 1, 2].map(c => {
      if (ops.random() < 0.5) { [a[c], b[c]] = [b[c], a[c]]; return true; }
      return false;
    });
    $('#lab-x-out').innerHTML = `<p>Crossed with a random partner: channels ${swapped.map((s, c) => (s ? 'RGB'[c] : '')).join('') || 'none'}
      swapped. The partner's child:</p>${list(b, (g, c) => `<code class="expr"><b>${'RGB'[c]}</b> ${esc(g)}</code>`)}`;
    set(a);
  });
  $('#lab-png').addEventListener('click', () => exportPng($('#lab-img'), 'basic_genome.png'));
  newGenome();
  mountEnhancedLab($('#lab-enh'));
});
