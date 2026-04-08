# Evolutionary Art System -- Roadmap

## Current State
A Python package (`evolutionary_art/`) with 5 modules totaling ~500 lines: core.py (basic GA), fitness.py (aesthetic evaluation), rendering.py (enhanced art with layer composition), styles.py (moods and style-specific functions), and cli.py (argparse CLI). Uses numpy, Pillow, DEAP, scipy, and optionally OpenCV. Has 12 pytest smoke tests. Supports reproducible generation via --seed flag.

## Short-term Improvements
- [x] Create a package structure: `evolutionary_art/{core.py, fitness.py, rendering.py, styles.py, cli.py}`
- [x] Extract shared logic between the two scripts: expression generation, evolution loop, image saving
- [x] Move the basic system's code into the package and make the enhanced system import from it
- [x] Add `requirements.txt` with numpy, Pillow, deap, matplotlib, scipy, opencv-python
- [x] Add CLI with argparse: `python -m evolutionary_art --style geometric --mood calm --generations 50 --size 800`
- [ ] Add type hints to fitness functions and expression generators
- [x] Add seed control for reproducible art generation (currently uses `random` without consistent seeding)

## Feature Enhancements
- [ ] Add interactive evolution: display a grid of candidates and let the user pick favorites (Tkinter or web UI)
- [ ] Add more artistic styles beyond geometric/organic/fluid/structured: watercolor, pointillist, glitch, fractal
- [ ] Add animation mode: interpolate between generations to create evolution timelapses (GIF/MP4 export)
- [ ] Add color palette presets (sunset, ocean, forest, neon) instead of random color generation
- [ ] Add high-resolution export (4K+) with tiled rendering for memory efficiency
- [ ] Add a fitness function based on neural style transfer or CLIP similarity to a text prompt
- [ ] Add multi-objective optimization: balance complexity, color harmony, and symmetry simultaneously

## Long-term Vision
- [ ] Add a web gallery that automatically publishes evolved art with metadata (parameters, lineage)
- [ ] Implement neural network-based aesthetic evaluation trained on user preferences
- [ ] Add 3D art evolution using signed distance functions or voxel grids
- [ ] Add collaborative evolution: multiple users vote on candidates in a shared population
- [ ] Integrate with generative AI: use evolved math expressions as control signals for diffusion models
- [ ] Add sound-reactive evolution: audio input influences mutation rates and selection pressure

## Technical Debt
- [x] `enhanced-art-system.py` (354 lines) duplicates the evolution loop from `evolutionary-art.py` -- violates DRY
- [x] Both scripts use global state and top-level execution -- wrap in proper `main()` functions with `if __name__`
- [ ] Expression generation uses string-based function building -- fragile and hard to debug; consider an AST approach
- [x] No error handling for degenerate expressions (division by zero, overflow in numpy operations)
- [x] `evolution_output/` directory creation is not guarded -- fails silently or overwrites previous runs
- [ ] DEAP dependency may be heavy for what's used -- evaluate if a simpler custom GA would suffice
