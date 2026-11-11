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
