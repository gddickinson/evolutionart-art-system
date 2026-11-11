"""Reference numbers from the real Python package for the JavaScript port (node tests and the Verify page).

Every case seeds Python's `random` and `np.random` exactly as the package does, runs the package's own code, and
records what came out: random streams, expression strings, palettes, rendered pixels, fitness terms, OpenCV filter
outputs and short seeded evolution runs of both systems. `scripts/make_js_fixtures.py` writes `fixture()` to
tests/fixtures/evo_reference.json; the JS tests replay every case and compare.
"""
import random

import numpy as np

from evolutionary_art.core import (
    ArtisticFunction, EvolutionaryArtSystem, generate_color_palettes, render_individual,
)
from evolutionary_art.fitness import evaluate_composition_thirds
from evolutionary_art.rendering import HAS_CV2, EnhancedEvolutionaryArt
from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction

SMALL = (20, 16)  # width, height of rendered fixtures (non-square on purpose)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _flat(a):
    return [float(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def rng_case():
    random.seed(12345)
    py = {
        "random": [random.random() for _ in range(6)],
        "randint_2_4": [random.randint(2, 4) for _ in range(8)],
        "choice": [random.choice(["a", "b", "c", "d", "e"]) for _ in range(8)],
        "uniform": [random.uniform(-2, 2) for _ in range(4)],
        "sample": [random.sample(list(range(20)), 3) for _ in range(4)],
        "sample5_k": [random.sample(["sin", "cos", "wave", "flow", "turbulence"], k=random.randint(1, 3)) for _ in range(4)],
    }
    np.random.seed(12345)
    npr = {"uniform": _flat(np.random.uniform(0.5, 1.5, 3)), "normal": _flat(np.random.normal(0, 0.1, 7))}
    floats = [0.5, -1.25, 2.0, 1e-05, 0.000123, 1.5e-7, 123456.789, 1e16, 0.1 + 0.2]
    return {"seed": 12345, "py": py, "np": npr, "float_repr": [[f, repr(f)] for f in floats]}


def basic_case():
    functions = []
    for seed in range(8):
        random.seed(seed)
        functions.append(ArtisticFunction().function_str)
    mutations = []
    for seed in range(40):
        random.seed(seed)
        f = ArtisticFunction()
        before = f.function_str
        random.seed(1000 + seed)
        f.mutate()
        mutations.append({"seed": 1000 + seed, "before": before, "after": f.function_str})
    random.seed(5)
    palettes = generate_color_palettes(5)

    w, h = SMALL
    system = EvolutionaryArtSystem(width=w, height=h, seed=4)
    system.initialize_population()
    individuals = []
    for ind in system.artistic_functions[:4]:
        img = render_individual(ind, w, h)
        parts = {k: float(v) for k, v in system.fitness_components(img).items()}
        individuals.append({"funcs": [f.function_str for f in ind], "pixels": _flat(img), "parts": parts})

    run_system = EvolutionaryArtSystem(width=w, height=h, seed=11)
    run_system.generations = 5
    run = []
    for gen, population, best in run_system.evolve_steps():
        run.append({"generation": gen, "fitness": [float(i.fitness.values[0]) for i in population],
                    "best": [f.function_str for f in best], "best_fitness": float(best.fitness.values[0])})
    return {"functions": functions, "mutations": mutations, "palettes": palettes, "size": [w, h],
            "system": {"seed": 4, "palettes": system.color_palettes, "individuals": individuals},
            "run": {"seed": 11, "generations": run}}


def _test_image(w, h, noise=0.15):
    """A smooth colour pattern with seeded noise: input for the OpenCV filter fixtures."""
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    img = np.stack([0.5 + 0.5 * np.sin(4 * x + y), 0.5 + 0.5 * np.cos(3 * y - x * x), np.abs(x * y)], axis=2)
    img += np.random.RandomState(9).uniform(-noise, noise, img.shape)
    return np.clip(img, 0, 1)


def cv_case():
    import cv2
    w, h = 24, 18
    img = _test_image(w, h)
    u8 = (img * 255).astype(np.uint8)
    return {
        "size": [w, h], "image": _flat(img),
        "canny": [int(v) for v in cv2.Canny(u8, 100, 200).ravel()],
        "canny_low": [int(v) for v in cv2.Canny(u8, 20, 60).ravel()],
        "smooth": _flat(_test_image(48, 36, noise=0.0)),
        "canny_smooth": [int(v) for v in cv2.Canny((_test_image(48, 36, noise=0.0) * 255).astype(np.uint8), 30, 60).ravel()],
        "blur15": _flat(cv2.GaussianBlur(img, (15, 15), 0)),
        "blur31": _flat(cv2.GaussianBlur(img, (31, 31), 0)),
        "bilateral": _flat(cv2.bilateralFilter(img.astype(np.float32), 9, 75, 75)),
        "thirds": float(evaluate_composition_thirds(img)),
        "gradient_mean": float(np.mean(np.abs(np.gradient(img)))),
    }


def enhanced_case():
    intents = []
    for seed, mood, style in [(1, None, None), (2, "calm", None), (3, None, "fluid"), (4, "energetic", "organic")]:
        random.seed(seed)
        it = ArtisticIntent(mood=mood, style=style)
        intents.append({"seed": seed, "mood_in": mood, "style_in": style, "mood": it.mood, "style": it.style,
                        "color_theme": [list(c) for c in it.color_theme], "composition_weight": it.composition_weight})
    expressions = []
    for seed, mood, style in [(21, "calm", "organic"), (22, "energetic", "fluid"), (23, "harmonious", "geometric")]:
        intent = ArtisticIntent(mood=mood, style=style)
        func = EnhancedArtisticFunction(intent)
        random.seed(seed)
        expressions.append({"seed": seed, "mood": mood, "style": style, "functions": func.functions,
                            "exprs": [func.generate_expression() for _ in range(4)]})
    w, h = SMALL
    artworks, runs = [], []
    for k, style in enumerate(ArtisticIntent.STYLES):
        art = EnhancedEvolutionaryArt(width=w, height=h, seed=30 + k, style=style)
        img = art.create_artwork([EnhancedArtisticFunction(art.intent)])
        score = art.evaluate_aesthetics(img)
        artworks.append({"seed": 30 + k, "style": style, "mood": art.intent.mood, "pixels": _flat(img),
                         "score": float(score)})
        art = EnhancedEvolutionaryArt(width=w, height=h, seed=40 + k, style=style)
        art.generations = 3
        gens = [{"generation": s["generation"], "scores": [float(v) for v in s["scores"]], "best_idx": s["best_idx"],
                 "mutation_rate": s["mutation_rate"], "functions": [p.functions for p in s["population"]],
                 "best_pixels": [int(v) for v in (s["artworks"][s["best_idx"]] * 255).astype(np.uint8).ravel()]}
                for s in art.evolve_steps()]
        runs.append({"seed": 40 + k, "style": style, "mood": art.intent.mood, "generations": gens})
    tables = {"moods": ArtisticIntent.MOODS, "styles": ArtisticIntent.STYLES,
              "base_functions": EnhancedArtisticFunction.BASE_FUNCTIONS,
              "style_functions": EnhancedArtisticFunction.STYLE_FUNCTIONS,
              "mood_functions": EnhancedArtisticFunction.MOOD_FUNCTIONS,
              "extra_functions": EnhancedEvolutionaryArt.EXTRA_FUNCTIONS,
              "base_colors": [list(c) for c in EnhancedEvolutionaryArt.BASE_COLORS]}
    return {"size": [w, h], "tables": tables, "intents": intents, "expressions": expressions, "artworks": artworks, "runs": runs}


def fixture():
    """Everything the JS tests compare against (needs OpenCV for the enhanced style effects)."""
    if not HAS_CV2:
        raise RuntimeError("the reference fixture needs OpenCV (opencv-python-headless)")
    return {"rng": rng_case(), "basic": basic_case(), "cv": cv_case(), "enhanced": enhanced_case()}
