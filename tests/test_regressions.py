"""Regression tests for bugs fixed in the existing package (one test per fix).

Run: python -m pytest -q tests/test_regressions.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolutionary_art.core import ArtisticFunction, EvolutionaryArtSystem  # noqa: E402


def test_basic_fitness_scores_sin_cos_expressions():
    """evaluate_individual() used a namespace without sin/cos, so every expression using them scored 0."""
    system = EvolutionaryArtSystem(width=16, height=16, seed=1)
    individual = [ArtisticFunction() for _ in range(3)]
    for func, expr in zip(individual, ["sin((1.5 * x))", "cos((0.7 * y))", "((0.3 * x) + (0.9 * y))"]):
        func.function_str = expr
    assert system.evaluate_individual(individual)[0] > 0
    system.initialize_population()
    scores = [system.evaluate_individual(ind)[0] for ind in system.artistic_functions]
    assert all(np.isfinite(scores)) and min(scores) > 0


def test_basic_offspring_do_not_share_genes(monkeypatch, tmp_path):
    """Offspring were shallow copies sharing ArtisticFunction objects, so mutating one child silently changed others
    whose stored fitness then no longer matched their genes (the 'best' image was saved with a stale score)."""
    import evolutionary_art.core as core
    saved = []
    monkeypatch.setattr(core, "save_artwork", lambda ind, *a, **k: saved.append(ind))
    stale = 0
    for seed in range(4):
        system = core.EvolutionaryArtSystem(width=16, height=16, seed=seed)
        system.generations = 8
        saved.clear()
        system.evolve_art(output_dir=str(tmp_path))
        stale += sum(abs(ind.fitness.values[0] - system.evaluate_individual(ind)[0]) > 1e-12 for ind in saved)
    assert stale == 0
