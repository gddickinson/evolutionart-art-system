"""FastAPI web app for the Evolutionary Art System: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8158, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The browser
runs both art systems itself (JS port in static/); POST /api/evolve runs the real Python package (local app only).
"""
import base64
import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from evolutionary_art.core import EvolutionaryArtSystem, render_individual
from evolutionary_art.rendering import HAS_CV2, EnhancedEvolutionaryArt
from evolutionary_art.styles import ArtisticIntent, EnhancedArtisticFunction

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLE_GENERATIONS = [0, 10, 25, 49]

app = FastAPI(title="Evolutionary Art System")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_lock = threading.Lock()  # the package uses the global random / np.random state


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


def _function_names():
    names = set(EnhancedArtisticFunction.BASE_FUNCTIONS) | set(EnhancedEvolutionaryArt.EXTRA_FUNCTIONS)
    for table in (EnhancedArtisticFunction.STYLE_FUNCTIONS, EnhancedArtisticFunction.MOOD_FUNCTIONS):
        for funcs in table.values():
            names |= set(funcs)
    return sorted(names)


@app.get("/api/info")
def info():
    """Project summary and counts for the home page (all from the package)."""
    return {
        "title": "Evolutionary Art System",
        "summary": "Genetic algorithms that create abstract visual art that evolves over time.",
        "systems": 2, "moods": len(ArtisticIntent.MOODS), "styles": len(ArtisticIntent.STYLES),
        "functions": len(_function_names()), "population_size": EnhancedEvolutionaryArt().population_size,
        "generations_default": EnhancedEvolutionaryArt().generations, "examples": len(EXAMPLE_GENERATIONS),
    }


@app.get("/api/options")
def options():
    """Moods, styles and the function-set tables used by the enhanced system."""
    return {
        "moods": ArtisticIntent.MOODS, "styles": ArtisticIntent.STYLES, "functions": _function_names(),
        "base_functions": EnhancedArtisticFunction.BASE_FUNCTIONS,
        "style_functions": EnhancedArtisticFunction.STYLE_FUNCTIONS,
        "mood_functions": EnhancedArtisticFunction.MOOD_FUNCTIONS,
        "extra_functions": EnhancedEvolutionaryArt.EXTRA_FUNCTIONS,
        "base_colors": [list(c) for c in EnhancedEvolutionaryArt.BASE_COLORS],
        "basic_operators": ["+", "*", "sin", "cos"], "has_cv2": HAS_CV2,
    }


@app.get("/api/examples")
def examples():
    """Thumbnails of images saved in evolution_output/ (made by scripts/make_example_thumbs.py)."""
    return [{"generation": g, "src": f"static/img/generation_{g}.jpg", "original": f"evolution_output/generation_{g}.png"}
            for g in EXAMPLE_GENERATIONS]


@lru_cache(maxsize=1)
def _reference():
    from evolutionary_art.web.reference import fixture
    return fixture()


@app.get("/api/reference")
def reference():
    """Seeded reference outputs of the Python package (the JS port is tested against the same numbers)."""
    return _reference()


class EvolveRequest(BaseModel):
    mode: str = Field("enhanced", pattern="^(basic|enhanced)$")
    seed: int = Field(1, ge=0, le=2**31 - 1)
    size: int = Field(64, ge=8, le=160)
    generations: int = Field(4, ge=1, le=12)
    mood: str | None = None
    style: str | None = None


def _png(img):
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/evolve")
def evolve(req: EvolveRequest):
    """Run the real Python system for a few generations and return each generation's best image and scores."""
    if req.mood and req.mood not in ArtisticIntent.MOODS or req.style and req.style not in ArtisticIntent.STYLES:
        raise HTTPException(422, "unknown mood or style")
    out = []
    with _lock:
        if req.mode == "basic":
            system = EvolutionaryArtSystem(width=req.size, height=req.size, seed=req.seed)
            system.generations = req.generations
            for gen, population, best in system.evolve_steps():
                out.append({"generation": gen, "scores": [float(i.fitness.values[0]) for i in population],
                            "best_score": float(best.fitness.values[0]), "genome": [f.function_str for f in best],
                            "png": _png(render_individual(best, req.size, req.size))})
            return {"mode": "basic", "palettes": system.color_palettes, "generations": out}
        art = EnhancedEvolutionaryArt(width=req.size, height=req.size, seed=req.seed, mood=req.mood, style=req.style)
        art.generations = req.generations
        for s in art.evolve_steps():
            b = s["best_idx"]
            out.append({"generation": s["generation"], "scores": [float(v) for v in s["scores"]],
                        "best_score": float(s["scores"][b]), "mutation_rate": s["mutation_rate"],
                        "genome": s["population"][b].functions, "png": _png(s["artworks"][b])})
        intent = {"mood": art.intent.mood, "style": art.intent.style, "color_theme": art.intent.color_theme}
        return {"mode": "enhanced", "intent": intent, "generations": out}


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/options", "/examples", "/reference"]
