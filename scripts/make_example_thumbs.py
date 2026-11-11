"""Write small JPEG thumbnails of a few saved generations in evolution_output/ for the web site's home page.

    python scripts/make_example_thumbs.py
The originals (800x800 PNG, ~0.6 MB each) stay where they are; the site only ships these ~20 kB copies.
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GENERATIONS = [0, 10, 25, 49]
OUT = ROOT / "evolutionary_art" / "web" / "static" / "img"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for g in GENERATIONS:
        im = Image.open(ROOT / "evolution_output" / f"generation_{g}.png").convert("RGB").resize((240, 240), Image.LANCZOS)
        im.save(OUT / f"generation_{g}.jpg", quality=85)
    print("wrote", ", ".join(f"generation_{g}.jpg" for g in GENERATIONS))


if __name__ == "__main__":
    main()
