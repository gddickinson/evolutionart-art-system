// node basic_engine.test.mjs: the basic system (core.py port) vs tests/fixtures/evo_reference.json.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PyRandom, NpRandom } from './py_random.js';
import { generateFunction, mutateFunction, generateColorPalettes, renderIndividual, BasicSystem } from './basic_engine.js';

const F = JSON.parse(readFileSync(new URL('../../../tests/fixtures/evo_reference.json', import.meta.url))).basic;
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);
const rng = seed => ({ py: new PyRandom(seed), np: new NpRandom(seed) });

F.functions.forEach((s, seed) => assert.equal(generateFunction(new PyRandom(seed)), s, `function seed ${seed}`));
for (const [i, m] of F.mutations.entries()) {
  assert.equal(generateFunction(new PyRandom(i)), m.before);
  assert.equal(mutateFunction(new PyRandom(m.seed), m.before), m.after, `mutation ${i}`);
}
assert.deepEqual(generateColorPalettes(new PyRandom(5), 5), F.palettes, 'palettes');

const [w, h] = F.size;
const sys = new BasicSystem(rng(F.system.seed), { width: w, height: h });
assert.deepEqual(sys.palettes, F.system.palettes, 'system palettes');
sys.initialize();
let maxPix = 0;
F.system.individuals.forEach((ref, k) => {
  assert.deepEqual(sys.population[k].funcs, ref.funcs, `initial individual ${k}`);
  const img = renderIndividual(ref.funcs, w, h);
  img.data.forEach((v, i) => { maxPix = Math.max(maxPix, Math.abs(v - ref.pixels[i])); });
  const parts = sys.fitnessComponents(img);
  for (const key of Object.keys(ref.parts)) close(parts[key], ref.parts[key], 1e-9, `${key} ${k}`);
});
assert.ok(maxPix < 1e-9, `pixels differ by ${maxPix}`);

const run = new BasicSystem(rng(F.run.seed), { width: w, height: h });
run.initialize();
for (const g of F.run.generations) {
  const s = run.step();
  assert.equal(s.generation, g.generation);
  s.population.forEach((ind, i) => close(ind.fitness, g.fitness[i], 1e-9, `gen ${g.generation} fitness ${i}`));
  assert.deepEqual(s.best.funcs, g.best, `gen ${g.generation} best genome`);
}
console.log(`basic_engine: ok (${F.functions.length} genomes, ${F.mutations.length} mutations, palettes, pixels max diff ${maxPix.toExponential(1)}, ${F.run.generations.length}-generation run)`);
