// node py_random.test.mjs: CPython random / numpy np.random ports vs tests/fixtures/evo_reference.json.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PyRandom, NpRandom, pyFloatStr } from './py_random.js';

const F = JSON.parse(readFileSync(new URL('../../../tests/fixtures/evo_reference.json', import.meta.url))).rng;
const py = new PyRandom(F.seed);
assert.deepEqual(Array.from({ length: 6 }, () => py.random()), F.py.random, 'random()');
assert.deepEqual(Array.from({ length: 8 }, () => py.randint(2, 4)), F.py.randint_2_4, 'randint');
assert.deepEqual(Array.from({ length: 8 }, () => py.choice(['a', 'b', 'c', 'd', 'e'])), F.py.choice, 'choice');
assert.deepEqual(Array.from({ length: 4 }, () => py.uniform(-2, 2)), F.py.uniform, 'uniform');
assert.deepEqual(Array.from({ length: 4 }, () => py.sample([...Array(20).keys()], 3)), F.py.sample, 'sample');
assert.deepEqual(Array.from({ length: 4 }, () => { const k = py.randint(1, 3); return py.sample(['sin', 'cos', 'wave', 'flow', 'turbulence'], k); }),
  F.py.sample5_k, 'sample with randint k');
const np = new NpRandom(F.seed);
assert.deepEqual([...np.uniform(0.5, 1.5, 3)], F.np.uniform, 'np uniform');
// log/sqrt of the polar method may differ in the last bit between V8 and the platform libm
[...np.normal(0, 0.1, 7)].forEach((v, i) => assert.ok(Math.abs(v - F.np.normal[i]) <= 1e-12, 'np normal'));
for (const [v, s] of F.float_repr) assert.equal(pyFloatStr(v), s, `repr(${v})`);
console.log('py_random: ok (MT streams bit-identical, normal within 1e-12, float repr)');
