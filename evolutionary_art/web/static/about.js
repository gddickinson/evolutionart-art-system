// About (#/about): what the two systems do (as the code implements them), how to run the desktop and web apps, credits.
import { route, app, api, list, esc } from './core.js';

route('about', async () => {
  const o = await api('/options');
  app().innerHTML = `
  <h1 class="glow">About</h1>
  <p class="lead">The Evolutionary Art System is a Python package with two generative art systems that use genetic
  algorithms and mathematical functions to create abstract visual art that evolves over time. This site runs both of
  them in the browser through a JavaScript port that reproduces the Python results seed for seed
  (<a href="#/verify">see the comparison</a>).</p>

  <h2>Basic system</h2>
  <p><b>Genome:</b> three random expressions in <code>x</code> and <code>y</code> built from
  <code>${o.basic_operators.join(' ')}</code>, one per colour channel, each normalised to 0..1.
  <b>Fitness:</b> 0.3 × complexity (standard deviation of the image) + 0.3 × left-right balance + 0.4 × colour harmony
  (how close the image comes to each colour of five random three-colour palettes).
  <b>Evolution</b> (DEAP): 20 individuals; tournament selection of size 3; crossover with probability 0.7 swaps each
  channel's expression with probability 1/2; mutation with probability 0.2 (then 0.2 per channel) either draws a new
  expression or scales coefficients by 0.5 to 1.5. The best individual of each generation is saved.</p>

  <h2>Enhanced system</h2>
  <p>An <b>artistic intent</b> fixes a mood (${list(o.moods, esc, ', ')}), a style (${list(o.styles, esc, ', ')}), a
  colour theme of two or three hues and a composition weight. <b>Genome:</b> a set of function names
  (${list(o.functions, f => `<code>${esc(f)}</code>`, ' ')}) chosen by style and mood. Each render rotates and scales the
  coordinate grid, draws three random expressions from the set, blends them (average, multiply or difference), tints the
  result with one of six base colours, and applies a style effect: edge overlay (geometric), noise and bilateral filter
  (organic), blur and sharpen (fluid), none (structured). <b>Fitness:</b> complexity (intensity spread, gradients and
  Canny edges; ×1.5 when energetic), rule-of-thirds composition and closeness to the colour theme, weighted by style and
  multiplied by a random 0.95 to 1.05. <b>Evolution:</b> parents by tournaments of 3, the child gets the union of their
  function sets, a mutation rate that starts at 0.2 and rises by 0.1 per generation without improvement (up to 0.8)
  adds 1 to 3 extra functions, and two fresh individuals join every generation.</p>

  <h2>Run it</h2>
  <h3>Command line (the Python package)</h3>
  <pre>pip install -r requirements.txt
python -m evolutionary_art --mode enhanced --style geometric --mood calm --generations 50 --size 800 --seed 1
python -m evolutionary_art --mode basic --generations 50 --output-dir evolution_output</pre>
  <p>Each generation's best image is written to <code>evolution_output/generation_N.png</code>.</p>
  <h3>This web app, locally</h3>
  <pre>pip install -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8158</pre>
  <p>The local app adds one thing the static site cannot do: running the Python package itself, on the
  <a href="#/verify">Python vs JS</a> page.</p>

  <h2>Notes on this port</h2>
  <ul>
    <li>The style effects are ports of the OpenCV functions the package calls (Canny, GaussianBlur, bilateralFilter);
    the package skips them when OpenCV is not installed, the browser always applies them.</li>
    <li>Four bugs were fixed in the Python package while building this site: basic-mode fitness was 0 for every
    expression using sin or cos, offspring shared genes so mutations leaked between them, the geometric style crashed,
    and <code>--seed</code> did not reproduce enhanced-mode art between runs.</li>
  </ul>

  <h2>Credits</h2>
  <p>By George Dickinson; created with assistance from Anthropic's Claude. MIT licence. Inspired by natural evolution
  and artistic processes, and by generative art pioneers.
  <a href="https://github.com/gddickinson/evolutionart-art-system">Source on GitHub</a>.</p>`;
});
