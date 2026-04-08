# Evolutionary Art System -- Interface Map

## Package: evolutionary_art/

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports: ArtisticFunction, EvolutionaryArtSystem, ArtisticIntent, EnhancedArtisticFunction, EnhancedEvolutionaryArt |
| `__main__.py` | Enables `python -m evolutionary_art` |
| `core.py` | ArtisticFunction (expression gen/mutation), EvolutionaryArtSystem (basic GA), save_artwork() |
| `fitness.py` | evaluate_color_harmony(), evaluate_complexity(), evaluate_balance(), evaluate_composition_thirds() |
| `rendering.py` | EnhancedEvolutionaryArt: layer composition, style effects, evolution loop |
| `styles.py` | ArtisticIntent (mood/style/color), EnhancedArtisticFunction (style-aware math ops) |
| `cli.py` | CLI: --mode, --style, --mood, --generations, --size, --seed, --output-dir |

## Top-Level Files

| File | Purpose |
|------|---------|
| `test_evolutionary_art.py` | Pytest smoke tests (12 tests) |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Ignores evolution_output/, __pycache__/, etc. |

## Data Flow

```
cli.py  -->  core.py (basic mode)
         \-> rendering.py (enhanced mode)
              |-> styles.py (mood/style config)
              |-> fitness.py (aesthetic scoring)
```

## Archive

- `_archive/evolutionary-art.py` -- original basic script
- `_archive/enhanced-art-system.py` -- original enhanced script
