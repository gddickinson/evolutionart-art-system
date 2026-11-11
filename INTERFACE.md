# Evolutionary Art System -- Interface Map

## Package: evolutionary_art/

| File | Purpose | Key names |
|------|---------|-----------|
| `__init__.py` | Package exports | ArtisticFunction, EvolutionaryArtSystem, ArtisticIntent, EnhancedArtisticFunction, EnhancedEvolutionaryArt |
| `__main__.py` | Enables `python -m evolutionary_art` | -- |
| `core.py` | Basic system: expression genomes, colour palettes, DEAP GA | `ArtisticFunction` (generate/mutate), `eval_function()`, `EXPRESSION_NAMES`, `generate_color_palettes()`, `EvolutionaryArtSystem` (`initialize_population`, `evaluate_individual`, `fitness_components`, `crossover`, `mutate_individual`, `evolve_steps()` generator, `evolve_art()`), `render_individual()`, `save_artwork()` |
| `fitness.py` | Aesthetic scoring | `evaluate_color_harmony()`, `evaluate_complexity()`, `evaluate_balance()`, `evaluate_composition_thirds()` |
| `styles.py` | Intent and function sets | `ArtisticIntent` (MOODS, STYLES, colour theme), `EnhancedArtisticFunction` (BASE_FUNCTIONS, STYLE_FUNCTIONS, MOOD_FUNCTIONS, math ops, `generate_expression()`) |
| `rendering.py` | Enhanced system | `EnhancedEvolutionaryArt` (BASE_COLORS, EXTRA_FUNCTIONS, `create_artwork`, `_apply_style_effects` (OpenCV), `evaluate_aesthetics`, `evolve_steps()` generator, `evolve()`) |
| `cli.py` | CLI: --mode, --style, --mood, --generations, --size, --seed, --output-dir | `main()` |

## Web: evolutionary_art/web/

| File | Purpose | Key names |
|------|---------|-----------|
| `app.py` | FastAPI app: static front end, `/api/info`, `/options`, `/examples`, `/reference` (exported for the static site), `POST /api/evolve` (runs the Python package; local app only) | `app`, `STATIC`, `STATIC_ENDPOINTS` |
| `reference.py` | Seeded reference runs of the package (RNG streams, genomes, mutations, pixels, fitness, OpenCV filters, short runs) | `fixture()`, `rng_case`, `basic_case`, `cv_case`, `enhanced_case` |
| `static/index.html`, `style.css` (kit), `style_app.css` | Shell, shared dark neon theme, magenta/amber/cyan accents | -- |
| `static/core.js` (kit), `main.js`, `static_api.js` | Router and `api()`, page registry, static-site API (exported JSON) | `staticApi()` |
| `static/py_random.js` | CPython `random` + numpy legacy `np.random` (MT19937), `repr(float)`, colorsys | `PyRandom`, `NpRandom`, `makeRng`, `pyFloatStr` |
| `static/expression.js` | Parser + vectorised evaluator for the expression strings; linspace/meshgrid grid | `parse`, `evaluate`, `evalExpression`, `grid`, `normalize` |
| `static/fitness.js` | Port of fitness.py + numpy mean/std/gradient | `evaluateColorHarmony`, `evaluateBalance`, `evaluateCompositionThirds`, `meanAbsGradient` |
| `static/cv_ops.js` | Ports of cv2.Canny, GaussianBlur, bilateralFilter | `canny`, `gaussianBlur`, `bilateralFilter`, `toUint8` |
| `static/basic_engine.js` | Port of core.py | `generateFunction`, `mutateFunction`, `renderIndividual`, `BasicSystem` |
| `static/enhanced_engine.js` | Port of styles.py + rendering.py | `makeIntent`, `functionSet`, `generateExpression`, `EnhancedArt` |
| `static/evolve_run.js`, `art_view.js` | Common run interface; canvas drawing, PNG export, history chart | `ArtRun`, `drawImage`, `exportPng`, `drawHistory` |
| `static/home.js`, `evolve.js`, `lab.js` (+`lab_enhanced.js`), `verify.js`, `about.js` | Pages `#/`, `#/evolve`, `#/lab`, `#/verify`, `#/about` | -- |
| `static/*.test.mjs` | Node tests of the JS port against `tests/fixtures/evo_reference.json` | -- |
| `static/img/` | Thumbnails of four saved generations from `evolution_output/` | -- |

## Top-Level Files

| File | Purpose |
|------|---------|
| `test_evolutionary_art.py` | Pytest smoke tests of the package (12 tests) |
| `tests/test_regressions.py` | Regression tests for the bugs fixed in Nov 2026 |
| `tests/test_web.py` | API endpoints, static build, fixture up to date, runs the node tests |
| `tests/fixtures/evo_reference.json` | Python reference outputs for the JS port (regenerate: `scripts/make_js_fixtures.py`) |
| `run_web.py` | Web app on http://127.0.0.1:8158 |
| `scripts/build_static.py` | Export the static GitHub Pages site to `site/` |
| `scripts/screenshots.py` | Playwright smoke test (`--static` / `--base`) and README images (`--shots` -> `docs/img/`) |
| `scripts/make_js_fixtures.py`, `scripts/make_example_thumbs.py` | Regenerate the JS fixture / home-page thumbnails |
| `.github/workflows/pages.yml` | Test, build and publish the static site on push to `main` |
| `requirements.txt`, `requirements-web.txt` | Package dependencies / pinned web app dependencies |
| `.gitignore` | Ignores site/, __pycache__/, .DS_Store |
| `evolution_output/` | 50 saved best images (800x800) from an earlier run, kept in git |

## Data Flow

```
cli.py  -->  core.py (basic mode: EvolutionaryArtSystem.evolve_steps -> save_artwork)
         \-> rendering.py (enhanced mode: EnhancedEvolutionaryArt.evolve_steps)
              |-> styles.py (mood/style, function sets, expressions)
              |-> fitness.py (aesthetic scoring)

Web: run_web.py -> evolutionary_art.web.app (/api/*; POST /evolve runs evolve_steps)
     browser: evolve.js -> ArtRun -> basic_engine.js / enhanced_engine.js (py_random, expression, fitness, cv_ops)
     scripts/build_static.py -> site/ (static/ + api/*.json) -> GitHub Pages
```

## Archive

- `_archive/evolutionary-art.py` -- original basic script
- `_archive/enhanced-art-system.py` -- original enhanced script
