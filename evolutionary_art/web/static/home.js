// Home page: hero with fresh renders from both systems, counts from the Python package, entry cards, saved examples.
import { route, api, app, list, esc } from './core.js';
import { makeRng } from './py_random.js';
import { EnhancedArt, STYLES } from './enhanced_engine.js';
import { BasicSystem, renderIndividual } from './basic_engine.js';
import { canvasFor } from './art_view.js';

const DOORS = [
  ['#/evolve', 'Evolve', 'Run either system generation by generation: the whole scored population, best and mean score, genomes, PNG export.', 'a'],
  ['#/lab', 'Genome lab', 'Edit a basic genome\'s R, G, B expressions, mutate and cross them over; take an enhanced render apart layer by layer.', 'b'],
  ['#/verify', 'Python vs JavaScript', 'Seeded runs of the Python package next to the browser port, with the differences measured.', 'c'],
  ['#/about', 'About', 'How the two systems work, what the fitness functions reward, and how to run the desktop and web apps.', 'a'],
];

function heroArt(host, seed) {
  host.innerHTML = '';
  STYLES.forEach((style, k) => {
    const art = new EnhancedArt(makeRng(seed + k), { width: 120, height: 120, style });
    const img = art.createArtwork([art.newIndividual()]);
    const a = document.createElement('a');
    a.href = `#/evolve?style=${style}&seed=${seed + k}`; a.className = 'hero-cell';
    a.appendChild(canvasFor(img, '', `Enhanced system, ${style} style, mood ${art.intent.mood}`));
    a.insertAdjacentHTML('beforeend', `<span>${esc(style)} · ${esc(art.intent.mood)}</span>`);
    host.appendChild(a);
  });
  const b = new BasicSystem(makeRng(seed), { width: 120, height: 120 });
  b.initialize();
  const a = document.createElement('a');
  a.href = `#/evolve?mode=basic&seed=${seed}`; a.className = 'hero-cell';
  a.appendChild(canvasFor(renderIndividual(b.population[0].funcs, 120, 120), '', 'Basic system render'));
  a.insertAdjacentHTML('beforeend', '<span>basic system</span>');
  host.appendChild(a);
}

route('home', async () => {
  const [info, examples] = await Promise.all([api('/info'), api('/examples')]);
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Genomes · Mutation · Selection</p>
    <h1 class="glow">Evolutionary Art System</h1>
    <p class="lead">${esc(info.summary)} Random mathematical expressions are rendered as images, scored by an aesthetic
    fitness function and bred over generations: both systems of the Python package, running live in your browser.</p>
    <a class="btn" href="#/evolve?run=1">Start evolving</a><a class="btn alt" href="#/about">How it works</a>
  </section>
  <div class="hero-art" id="hero-art"></div>
  <p class="small mute">Fresh first-generation renders (one per enhanced style, plus the basic system); click one to evolve
  it. <button type="button" class="alt" id="reroll">Render new</button></p>
  <div class="tiles">
    <div class="tile"><b>${info.systems}</b><span>art systems</span></div>
    <div class="tile"><b>${info.styles}</b><span>styles</span></div>
    <div class="tile"><b>${info.moods}</b><span>moods</span></div>
    <div class="tile"><b>${info.functions}</b><span>expression functions</span></div>
    <div class="tile"><b>${info.population_size}</b><span>individuals per generation</span></div>
  </div>
  <div class="grid wide">${list(DOORS, ([href, t, d, c]) => `<a class="card door ${c}" href="${href}"><b>${t}</b><span>${d}</span></a>`)}</div>
  <h2>From the repository</h2>
  <p class="small mute">Best images of generations ${examples.map(e => e.generation).join(', ')} of one earlier run, saved in
  <code>evolution_output/</code> (800 × 800 originals, shown reduced).</p>
  <div class="examples">${list(examples, e => `<figure><img src="${esc(e.src)}" alt="Saved best image of generation ${e.generation}" width="240" height="240" loading="lazy"><figcaption>generation ${e.generation}</figcaption></figure>`)}</div>`;
  let seed = 1 + Math.floor(Math.random() * 1000);
  const host = document.getElementById('hero-art');
  heroArt(host, seed);
  document.getElementById('reroll').addEventListener('click', () => { seed += 10; heroArt(host, seed); });
});
