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


def test_enhanced_every_style_renders():
    """The geometric style blended a 3-channel image with a 1-channel edge map via cv2.addWeighted, which raises."""
    import pytest
    from evolutionary_art.rendering import HAS_CV2, EnhancedEvolutionaryArt
    from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction
    if not HAS_CV2:
        pytest.skip("style effects need OpenCV")
    for style in ArtisticIntent.STYLES:
        art = EnhancedEvolutionaryArt(width=24, height=24, seed=0, style=style, mood="calm")
        img = art.create_artwork([EnhancedArtisticFunction(art.intent)])
        assert img.shape == (24, 24, 3) and np.isfinite(img).all() and 0 <= img.min() <= img.max() <= 1
        assert np.isfinite(art.evaluate_aesthetics(img))


def test_enhanced_seed_is_reproducible_across_processes():
    """Function sets were built with list(set(...)), whose order depends on Python's per-process string hashing, so
    --seed gave different art in every run. Two processes with different hash seeds must now agree."""
    import os
    import subprocess
    code = ("import numpy as np, hashlib\n"
            "from evolutionary_art.rendering import EnhancedEvolutionaryArt\n"
            "from evolutionary_art.styles import EnhancedArtisticFunction\n"
            "a = EnhancedEvolutionaryArt(width=16, height=16, seed=3, style='fluid', mood='calm')\n"
            "f = EnhancedArtisticFunction(a.intent)\n"
            "print(f.functions, hashlib.md5(a.create_artwork([f]).tobytes()).hexdigest())\n")
    outs = set()
    for hs in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1, outs


def test_basic_mutation_always_changes_genome():
    """mutate() only touched coefficients its '*'-split happened to isolate, so about half the calls changed nothing."""
    import random
    import re
    coef = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
    unchanged = scaled = 0
    for seed in range(300):
        random.seed(seed)
        func = ArtisticFunction()
        before = func.function_str
        random.seed(10_000 + seed)
        regenerate = random.random() < 0.3  # mutate()'s first draw picks the branch
        random.seed(10_000 + seed)
        func.mutate()
        unchanged += func.function_str == before
        if not regenerate:
            scaled += 1  # coefficient branch: same structure, only numbers changed
            assert coef.sub("N", func.function_str) == coef.sub("N", before)
            new, old = coef.findall(func.function_str), coef.findall(before)
            assert all(0.5 <= float(a) / float(b) <= 1.5 for a, b in zip(new, old) if a != b)
    assert unchanged == 0
    assert scaled > 150
