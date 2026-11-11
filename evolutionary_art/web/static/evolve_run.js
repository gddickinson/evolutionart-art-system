// One evolution run of either system behind a common interface for the pages: step() returns the scored
// population of a generation as { generation, items: [{ img, score, genome, parts }], bestIdx, mutationRate }.
import { makeRng } from './py_random.js';
import { BasicSystem, renderIndividual } from './basic_engine.js';
import { EnhancedArt } from './enhanced_engine.js';

export class ArtRun {
  constructor({ mode = 'enhanced', seed = 1, size = 96, mood = null, style = null } = {}) {
    Object.assign(this, { mode, seed, size });
    this.rng = makeRng(seed);
    this.history = [];
    this.last = null;
    if (mode === 'basic') {
      this.sys = new BasicSystem(this.rng, { width: size, height: size });
      this.sys.initialize();
    } else {
      this.sys = new EnhancedArt(this.rng, { width: size, height: size, mood: mood || null, style: style || null });
    }
  }

  get generation() { return this.last ? this.last.generation : -1; }

  step() {
    let res;
    if (this.mode === 'basic') {
      const s = this.sys.step();
      const items = s.population.map(ind => ({
        img: renderIndividual(ind.funcs, this.size, this.size), score: ind.fitness, genome: ind.funcs, parts: ind.parts,
      }));
      res = { generation: s.generation, items, bestIdx: s.population.indexOf(s.best), mutationRate: null };
    } else {
      const s = this.sys.step();
      const items = s.artworks.map((img, i) => ({ img, score: s.scores[i], genome: s.population[i].functions, parts: s.parts[i] }));
      res = { generation: s.generation, items, bestIdx: s.bestIdx, mutationRate: s.mutationRate };
    }
    const scores = res.items.map(i => i.score);
    this.history.push({ generation: res.generation, best: Math.max(...scores), mean: scores.reduce((a, b) => a + b, 0) / scores.length });
    this.last = res;
    return res;
  }

  // What the run is aiming at: the enhanced intent, or the basic system's colour palettes.
  describe() {
    return this.mode === 'basic' ? { palettes: this.sys.palettes } : { intent: this.sys.intent };
  }
}
