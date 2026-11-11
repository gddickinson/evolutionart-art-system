// node enhanced_engine.test.mjs: the enhanced system (styles.py + rendering.py port) vs tests/fixtures/evo_reference.json.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PyRandom, NpRandom } from './py_random.js';
import { MOODS, STYLES, BASE_FUNCTIONS, STYLE_FUNCTIONS, MOOD_FUNCTIONS, EXTRA_FUNCTIONS, BASE_COLORS, makeIntent, functionSet, generateExpression, EnhancedArt } from './enhanced_engine.js';

const F = JSON.parse(readFileSync(new URL('../../../tests/fixtures/evo_reference.json', import.meta.url))).enhanced;
const rng = seed => ({ py: new PyRandom(seed), np: new NpRandom(seed) });
const [w, h] = F.size;

assert.deepEqual({ moods: MOODS, styles: STYLES, base_functions: BASE_FUNCTIONS, style_functions: STYLE_FUNCTIONS,
  mood_functions: MOOD_FUNCTIONS, extra_functions: EXTRA_FUNCTIONS, base_colors: BASE_COLORS }, F.tables, 'tables');
for (const c of F.intents) {
  const it = makeIntent(new PyRandom(c.seed), c.mood_in, c.style_in);
  assert.deepEqual([it.mood, it.style, it.color_theme, it.composition_weight], [c.mood, c.style, c.color_theme, c.composition_weight]);
}
for (const c of F.expressions) {
  const intent = { mood: c.mood, style: c.style };
  assert.deepEqual(functionSet(intent), c.functions, 'function set');
  const py = new PyRandom(c.seed);
  assert.deepEqual(c.exprs.map(() => generateExpression(py, intent, c.functions)), c.exprs, `expressions seed ${c.seed}`);
}
// Single artworks per style. Organic runs through a float32 bilateral filter in OpenCV, so allow float32 rounding there.
for (const c of F.artworks) {
  const art = new EnhancedArt(rng(c.seed), { width: w, height: h, style: c.style });
  assert.equal(art.intent.mood, c.mood);
  const img = art.createArtwork([art.newIndividual()]);
  const d = img.data.reduce((m, v, i) => Math.max(m, Math.abs(v - c.pixels[i])), 0);
  const tol = c.style === 'organic' ? 1e-5 : 1e-9;
  assert.ok(d < tol, `${c.style} pixels differ by ${d}`);
  const score = art.evaluateAesthetics(img);
  assert.ok(Math.abs(score - c.score) < tol, `${c.style} score ${score} vs ${c.score}`);
  console.log(`  ${c.style.padEnd(10)} artwork max pixel diff ${d.toExponential(1)}, score ${score.toFixed(6)} vs ${c.score.toFixed(6)}`);
}
// Three-generation runs: same scores, best index, mutation rate, genomes and saved best PNG bytes (within 1 level).
for (const r of F.runs) {
  const art = new EnhancedArt(rng(r.seed), { width: w, height: h, style: r.style });
  const tol = r.style === 'organic' ? 1e-5 : 1e-9;
  for (const g of r.generations) {
    assert.deepEqual(art.population.map(p => p.functions), g.functions, `${r.style} gen ${g.generation} genomes`);
    const s = art.step();
    s.scores.forEach((v, i) => assert.ok(Math.abs(v - g.scores[i]) < tol, `${r.style} gen ${g.generation} score ${i}`));
    assert.equal(s.bestIdx, g.best_idx); assert.equal(s.mutationRate, g.mutation_rate);
    const best = s.artworks[s.bestIdx], px = best.data.map(v => Math.trunc(best.f32 ? Math.fround(v * 255) : v * 255));
    const off = px.reduce((m, v, i) => Math.max(m, Math.abs(v - g.best_pixels[i])), 0);
    assert.ok(off <= 1, `${r.style} best PNG differs by ${off} levels`);
  }
}
console.log(`enhanced_engine: ok (${F.intents.length} intents, ${F.expressions.length} expression streams, 4 styles, ${F.runs.length} seeded 3-generation runs)`);
