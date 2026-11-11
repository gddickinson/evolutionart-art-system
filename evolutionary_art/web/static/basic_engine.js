// Port of evolutionary_art/core.py (the basic system): ArtisticFunction expression genomes, colour palettes,
// rendering (one expression per R, G, B channel), fitness, and the DEAP loop (tournament selection of size 3,
// crossover 0.7 swapping channel expressions, mutation 0.2). Same random calls in the same order as Python, so a
// seed reproduces the Python run.
import { pyFloatStr, pyFloatParse, hsvToRgb } from './py_random.js';
import { evalExpression, grid, normalize, allFinite } from './expression.js';
import { makeImage, std, evaluateBalance, evaluateColorHarmony } from './fitness.js';

// core.ArtisticFunction: a random expression in x, y with +, *, sin, cos (depth 2-4).
export function generateFunction(py) { return subfunction(py, py.randint(2, 4)); }

function subfunction(py, depth) {
  if (depth <= 1) {
    const v = py.choice(['x', 'y']);
    const coef = py.uniform(-2, 2);
    return `(${pyFloatStr(coef)} * ${v})`;
  }
  if (py.random() < 0.4) {
    const f = py.choice(['sin', 'cos']);
    return `${f}(${subfunction(py, depth - 1)})`;
  }
  const op = py.choice(['+', '*']);
  const left = subfunction(py, depth - 1), right = subfunction(py, depth - 1);
  return `(${left} ${op} ${right})`;
}

const pyStrip = (s, chars) => { let a = 0, b = s.length; while (a < b && chars.includes(s[a])) a++; while (b > a && chars.includes(s[b - 1])) b--; return s.slice(a, b); };

// ArtisticFunction.mutate: 30% a new expression, else scale some coefficients by U(0.5, 1.5) via the same string edits.
export function mutateFunction(py, src) {
  if (py.random() < 0.3) return generateFunction(py);
  let out = src;
  for (const part of src.split('*')) {
    if (py.random() < 0.3 && part.includes('.')) {
      const words = part.trim().split(/\s+/).filter(Boolean);
      if (!words.length) continue;
      const num = pyFloatParse(pyStrip(words[0], '() '));
      if (Number.isNaN(num)) continue;
      const nn = num * py.uniform(0.5, 1.5);
      out = out.split(pyFloatStr(num)).join(pyFloatStr(nn));
    }
  }
  return out;
}

// core.generate_color_palettes: n palettes of 3 neighbouring hues, RGB 0-255.
export function generateColorPalettes(py, n = 5) {
  const palettes = [];
  for (let k = 0; k < n; k++) {
    const base = py.random(), pal = [];
    for (let i = 0; i < 3; i++) {
      const hue = (base + 0.1 * i) % 1.0, sat = py.uniform(0.5, 1.0), val = py.uniform(0.5, 1.0);
      pal.push(hsvToRgb(hue, sat, val).map(c => Math.trunc(c * 255)));
    }
    palettes.push(pal);
  }
  return palettes;
}

// core.render_individual: channel i = normalised expression i.
export function renderIndividual(funcs, w, h) {
  const { X, Y } = grid(w, h), img = makeImage(w, h);
  funcs.forEach((f, c) => {
    const ch = normalize(evalExpression(f, X, Y), 1e-10);
    for (let i = 0; i < ch.length; i++) img.data[i * 3 + c] = ch[i];
  });
  return img;
}

export class BasicSystem {
  // rng: { py: PyRandom, np: NpRandom } already seeded (EvolutionaryArtSystem seeds both, then draws palettes).
  constructor(rng, { width = 128, height = 128, populationSize = 20 } = {}) {
    Object.assign(this, { rng, py: rng.py, width, height, populationSize, generation: -1 });
    this.palettes = generateColorPalettes(this.py);
    this.population = [];
  }

  // EvolutionaryArtSystem.fitness_components
  fitnessComponents(img) {
    const complexity = std(img.data), balance = evaluateBalance(img), colorHarmony = evaluateColorHarmony(img, this.palettes);
    return { complexity, balance, color_harmony: colorHarmony, fitness: complexity * 0.3 + balance * 0.3 + colorHarmony * 0.4 };
  }

  evaluate(ind) {
    try {
      ind.parts = this.fitnessComponents(renderIndividual(ind.funcs, this.width, this.height));
      ind.fitness = ind.parts.fitness;
    } catch (e) { ind.parts = null; ind.fitness = 0; }
    return ind.fitness;
  }

  // initialize_population (keeps individuals whose three expressions are finite on a 10x10 grid), then evaluate.
  initialize() {
    const { X, Y } = grid(10, 10);
    this.population = [];
    while (this.population.length < this.populationSize) {
      const funcs = [0, 1, 2].map(() => generateFunction(this.py));
      let ok = true;
      for (const f of funcs) { try { if (!allFinite(evalExpression(f, X, Y))) { ok = false; break; } } catch (e) { ok = false; break; } }
      if (ok) this.population.push({ funcs, fitness: null });
    }
    this.population.forEach(ind => this.evaluate(ind));
    this.generation = -1;
    return this.population;
  }

  // deap.tools.selTournament(population, k, tournsize=3): max of 3 random.choice picks (first max wins ties).
  selTournament(pop, k, size = 3) {
    const chosen = [];
    for (let i = 0; i < k; i++) {
      const asp = Array.from({ length: size }, () => this.py.choice(pop));
      chosen.push(asp.reduce((best, a) => (a.fitness > best.fitness ? a : best)));
    }
    return chosen;
  }

  // One generation of evolve_steps(); returns { generation, population, best }.
  step() {
    const py = this.py;
    const off = this.selTournament(this.population, this.population.length)
      .map(ind => ({ funcs: [...ind.funcs], fitness: ind.fitness, parts: ind.parts }));
    for (let i = 0; i + 1 < off.length; i += 2) {
      if (py.random() < 0.7) {
        const a = off[i], b = off[i + 1];
        for (let c = 0; c < a.funcs.length; c++) if (py.random() < 0.5) [a.funcs[c], b.funcs[c]] = [b.funcs[c], a.funcs[c]];
        a.fitness = null; b.fitness = null;
      }
    }
    for (const m of off) {
      if (py.random() < 0.2) {
        m.funcs = m.funcs.map(f => (py.random() < 0.2 ? mutateFunction(py, f) : f));
        m.fitness = null;
      }
    }
    for (const ind of off) if (ind.fitness === null) this.evaluate(ind);
    this.population = off;
    this.generation += 1;
    const best = off.reduce((b, a) => (a.fitness > b.fitness ? a : b));
    return { generation: this.generation, population: off, best };
  }
}
