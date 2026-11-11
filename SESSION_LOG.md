# Session log

## 2026-11-11 -- web interface and static GitHub Pages site (branch `web-interface`)

**Built** (web kit, see `INTERFACE.md`): FastAPI app `evolutionary_art/web/app.py` (`python run_web.py`, port 8158) and
a static site in which both systems run in the browser through a JavaScript port: CPython `random` and numpy's legacy
`np.random` (MT19937), the expression genomes, mutation/crossover/selection, fitness, and ports of the three OpenCV
functions used (Canny, GaussianBlur, bilateralFilter). Pages: Home, Evolve, Genome lab, Python vs JS, About.
The Python package has no interactive selection, so the site has none either (it is on ROADMAP.md).

**JS vs Python** (`tests/fixtures/evo_reference.json`, written by `scripts/make_js_fixtures.py` from
`evolutionary_art/web/reference.py`): random streams, genome strings, 40 mutations, palettes, function sets and Canny
edge maps identical; pixels and scores within ~1e-15; organic style (OpenCV float32 bilateral) within ~1e-7 per pixel,
and a live run showed one organic score differing by 1e-4 when a float32 rounding moved a pixel across an 8-bit level
used by the Canny term. Seeded 5-generation basic run and 3-generation enhanced runs (all 4 styles) reproduce Python.

**Refactors (no output change, verified by identical PNG bytes for CLI runs before/after):** `evolve_steps()` generators
in core.py and rendering.py (`evolve_art()` / `evolve()` consume them), `render_individual()`, `fitness_components()`,
`eval_function()`, function-set tables as class constants (`STYLE_FUNCTIONS`, `MOOD_FUNCTIONS`, `EXTRA_FUNCTIONS`).

**Bugs fixed** (each in its own commit, regression tests in `tests/test_regressions.py`):
1. Basic fitness: expressions were evaluated without `sin`/`cos` in scope, so they all scored 0.
2. Basic GA: offspring were shallow copies sharing genes; mutations leaked into siblings with stale fitness.
3. Geometric style crashed (`cv2.addWeighted` with a 3-channel image and a 1-channel edge map).
4. `--seed` did not reproduce enhanced art: function sets came from `list(set(...))` (hash-randomised order).
Docs: README usage named a non-existent `EvolutionaryArt` class and claimed "MIDI-like smooth transitions"; corrected.
`.gitignore` added (INTERFACE.md said one existed); `.DS_Store` files untracked.

**Open for George:** `evolution_output/` (50 PNGs, ~28 MB) is committed; the site uses four 240 px thumbnails.
Whether to keep it in git is your call. The basic `mutate()` only perturbs coefficients that its string split happens
to isolate (about half the time nothing changes); ported as is, not "fixed".

**Tests:** `python -m pytest -q tests/ test_evolutionary_art.py` (30 passed, includes 4 node test files).
