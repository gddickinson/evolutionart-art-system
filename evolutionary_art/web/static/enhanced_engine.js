// Port of the enhanced system: styles.py (ArtisticIntent, EnhancedArtisticFunction) and rendering.py
// (EnhancedEvolutionaryArt: create_artwork, style effects, evaluate_aesthetics, evolve_steps). The genome is a list
// of function names; every render draws a fresh random expression, rotation, scale, shifts and blend from it, using
// Python's random and numpy's np.random in the same order as the Python code.
import { pyFloatStr, hsvToRgb, pmod } from './py_random.js';
import { parse, evaluate, grid, normalize, allFinite } from './expression.js';
import { makeImage, mean, std, evaluateCompositionThirds, meanAbsGradient } from './fitness.js';
import { canny, toUint8, gaussianBlur, bilateralFilter } from './cv_ops.js';

export const MOODS = ['calm', 'energetic', 'mysterious', 'harmonious'];
export const STYLES = ['geometric', 'organic', 'fluid', 'structured'];
export const BASE_FUNCTIONS = ['sin', 'cos', 'wave'];
export const STYLE_FUNCTIONS = {
  geometric: ['abs', 'sin', 'cos'], organic: ['sigmoid', 'smooth_abs', 'flow'],
  fluid: ['flow', 'wave', 'turbulence'], structured: ['sin', 'cos', 'abs'],
};
export const MOOD_FUNCTIONS = {
  energetic: ['turbulence', 'wave'], calm: ['smooth_abs', 'flow'], mysterious: ['flow', 'wave'], harmonious: ['sin', 'cos'],
};
export const EXTRA_FUNCTIONS = ['sin', 'cos', 'wave', 'flow', 'turbulence'];
export const BASE_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]];
const STYLE_WEIGHTS = { organic: [0.4, 0.3, 0.3], geometric: [0.3, 0.4, 0.3], fluid: [0.35, 0.35, 0.3] };

const sortedSet = a => [...new Set(a)].sort();   // Python sorted(set(...)) of str: code-point order

// styles.ArtisticIntent
export function makeIntent(py, mood = null, style = null) {
  mood = mood || py.choice(MOODS);
  style = style || py.choice(STYLES);
  const base = py.random();
  let theme;
  if (py.random() < 0.3) theme = [[base, 0.7, 0.9], [(base + 0.5) % 1.0, 0.7, 0.9]];
  else {
    const spread = py.uniform(0.05, 0.15);
    theme = [[base, 0.7, 0.9], [(base + spread) % 1.0, 0.7, 0.9], [pmod(base - spread, 1.0), 0.7, 0.9]];
  }
  return { mood, style, color_theme: theme, composition_weight: py.uniform(0.3, 0.7) };
}

// EnhancedArtisticFunction._build_function_set
export const functionSet = intent =>
  sortedSet([...BASE_FUNCTIONS, ...(STYLE_FUNCTIONS[intent.style] || []), ...(MOOD_FUNCTIONS[intent.mood] || [])]);

// generate_expression: depth 3 when energetic else 2; returns the Python expression string.
export function generateExpression(py, intent, functions) {
  const build = depth => {
    if (depth <= 0) { const c = py.uniform(-2, 2); return `(${pyFloatStr(c)} * ${py.choice(['x', 'y'])})`; }
    const f = py.choice(functions);
    if (f === 'sigmoid' || f === 'smooth_abs') return `self.${f}(${build(depth - 1)}, None)`;
    if (f === 'flow' || f === 'wave' || f === 'turbulence') return `self.${f}(x, y)`;
    return `np.${f}(${build(depth - 1)})`;
  };
  return build(intent.mood === 'energetic' ? 3 : 2);
}

export class EnhancedArt {
  // rng: { py, np } seeded as EnhancedEvolutionaryArt(seed=...) seeds random and np.random, before the intent is drawn.
  constructor(rng, { width = 128, height = 128, mood = null, style = null, populationSize = 20 } = {}) {
    Object.assign(this, { py: rng.py, np: rng.np, width, height, populationSize });
    this.intent = makeIntent(this.py, mood, style);
    this.reset();
  }

  newIndividual() { return { functions: functionSet(this.intent) }; }

  // create_artwork([individual]); also returns the drawn expressions and blend ops when trace is an array.
  createArtwork(individuals, trace = null) {
    const { width: w, height: h, py } = this, n = w * h, img = makeImage(w, h), d = img.data;
    const { X, Y } = grid(w, h);
    const theta = py.uniform(0, 2 * Math.PI), scale = py.uniform(0.5, 2.0);
    const ct = Math.cos(theta), st = Math.sin(theta), Xf = new Float64Array(n), Yf = new Float64Array(n);
    for (let i = 0; i < n; i++) { Xf[i] = (X[i] * ct - Y[i] * st) * scale; Yf[i] = (X[i] * st + Y[i] * ct) * scale; }
    for (const ind of individuals) {
      const layers = [], exprs = [];
      for (let l = 0; l < 3; l++) {
        const sx = py.uniform(-0.5, 0.5), sy = py.uniform(-0.5, 0.5), freq = py.uniform(1, 5);
        py.random();                                      // the unused "random" namespace entry
        const xs = Xf.map(v => v * freq + sx), ys = Yf.map(v => v * freq + sy);
        const expr = generateExpression(py, this.intent, ind.functions);
        exprs.push(expr);
        let r = evaluate(parse(expr), xs, ys);
        if (r instanceof Float64Array && allFinite(r)) layers.push(normalize(r, 1e-6));
      }
      const ops = [];
      if (layers.length) {
        let comb = layers[0];
        for (const layer of layers.slice(1)) {
          const op = py.choice(['add', 'multiply', 'difference']); ops.push(op);
          comb = comb.map((v, i) => (op === 'add' ? (v + layer[i]) / 2 : op === 'multiply' ? v * layer[i] : Math.abs(v - layer[i])));
        }
        const base = py.choice(BASE_COLORS), cv = this.np.uniform(0.5, 1.5, 3);
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) d[i * 3 + c] += comb[i] * base[c] * cv[c];
        if (trace) trace.push({ exprs, ops, base, variation: [...cv] });
      } else if (trace) trace.push({ exprs, ops, base: null, variation: null });
    }
    let mx = -Infinity; for (const v of d) if (v > mx) mx = v;
    if (mx > 0) for (let i = 0; i < d.length; i++) d[i] /= mx;
    const fx = this.applyStyleEffects(d);
    img.f32 = fx.f32;
    const out = img.data = fx.data;
    for (let i = 0; i < out.length; i++) {
      const v = fx.f32 ? Math.fround(Math.fround(Math.fround(out[i] - 0.5) * 1.5) + 0.5) : (out[i] - 0.5) * 1.5 + 0.5;
      out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return img;
  }

  // _apply_style_effects (OpenCV present). Organic returns float32 data, as cv2.bilateralFilter does in Python.
  applyStyleEffects(d) {
    const { width: w, height: h } = this, style = this.intent.style;
    if (style === 'geometric') {
      const edges = canny(toUint8(d), w, h, 3, 100, 200), out = new Float64Array(d.length);
      for (let i = 0; i < d.length; i++) out[i] = d[i] * 0.7 + (edges[Math.floor(i / 3)] / 255) * 0.3;
      return { data: out, f32: false };
    }
    if (style === 'fluid') {
      const b1 = gaussianBlur(d, w, h, 15), b2 = gaussianBlur(b1, w, h, 31);
      return { data: b1.map((v, i) => v * 1.5 + b2[i] * -0.5 + 0), f32: false };
    }
    if (style === 'organic') {
      const noise = this.np.normal(0, 0.1, d.length);
      const noisy = d.map((v, i) => Math.min(1, Math.max(0, v + noise[i])));
      return { data: Float64Array.from(bilateralFilter(noisy, w, h, 9, 75, 75)), f32: true };
    }
    return { data: d, f32: false };
  }

  // evaluate_aesthetics: complexity, rule-of-thirds composition and colour-theme harmony, weighted by style, x U(0.95, 1.05).
  evaluateAesthetics(img, parts = null) {
    const { w, h, data } = img;
    const ic = std(data), gc = meanAbsGradient(data, [h, w, 3]);
    const edges = canny(toUint8(data, img.f32), w, h, 3, 100, 200);
    let complexity = (ic + gc + mean(edges) / 255) / 3;
    if (this.intent.mood === 'energetic') complexity *= 1.5;
    const composition = evaluateCompositionThirds(img);
    const scores = this.intent.color_theme.map(([hh, ss, vv]) => {
      const rgb = hsvToRgb(hh, ss, vv), diff = new Float64Array(data.length), proj = new Float64Array(w * h);
      for (let i = 0; i < data.length; i++) diff[i] = Math.abs(data[i] - rgb[i % 3]);
      for (let i = 0; i < w * h; i++) proj[i] = data[i * 3] * rgb[0] + data[i * 3 + 1] * rgb[1] + data[i * 3 + 2] * rgb[2];
      return 1.0 - mean(diff) + meanAbsGradient(proj, [h, w]);
    });
    const harmony = mean(scores), wts = STYLE_WEIGHTS[this.intent.style] || [0.33, 0.33, 0.34];
    const score = (complexity * wts[0] + composition * wts[1] + harmony * wts[2]) * this.py.uniform(0.95, 1.05);
    if (parts) Object.assign(parts, { complexity, composition, color_harmony: harmony, score });
    return score;
  }

  reset() {
    this.population = Array.from({ length: this.populationSize }, () => this.newIndividual());
    this.bestScore = -Infinity; this.stale = 0; this.generation = -1;
  }

  // One generation of evolve_steps(): score the population, then breed the next one (tournament of 3 from
  // random.sample, union of the parents' functions, adaptive mutation rate, two fresh immigrants).
  step() {
    const py = this.py, pop = this.population, artworks = [], scores = [], parts = [];
    for (const ind of pop) {
      const art = this.createArtwork([ind]), p = {};
      artworks.push(art); scores.push(this.evaluateAesthetics(art, p)); parts.push(p);
    }
    const cur = Math.max(...scores);
    if (cur > this.bestScore) { this.bestScore = cur; this.stale = 0; } else this.stale += 1;
    const mutationRate = Math.min(0.8, 0.2 + this.stale * 0.1);
    const next = [], indexed = pop.map((ind, i) => [i, ind]);
    while (next.length < this.populationSize) {
      if (py.random() < 0.7) {
        const parents = [0, 1].map(() => {
          const cands = py.sample(indexed, 3);
          return pop[cands.reduce((b, c) => (scores[c[0]] > scores[b[0]] ? c : b))[0]];
        });
        const child = this.newIndividual();
        child.functions = sortedSet([...parents[0].functions, ...parents[1].functions]);
        if (py.random() < mutationRate) {
          const k = py.randint(1, 3);
          child.functions.push(...py.sample(EXTRA_FUNCTIONS, k));
        }
        next.push(child);
      } else next.push(this.newIndividual());
    }
    const numNew = Math.max(1, Math.floor(this.populationSize / 10));
    for (let i = next.length - numNew; i < next.length; i++) next[i] = this.newIndividual();
    this.population = next;
    this.generation += 1;
    let bestIdx = 0; scores.forEach((s, i) => { if (s > scores[bestIdx]) bestIdx = i; });
    return { generation: this.generation, artworks, scores, parts, bestIdx, mutationRate, population: pop };
  }
}
