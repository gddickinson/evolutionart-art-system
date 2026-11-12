"""Write small JPEG thumbnails of a few saved generations in evolution_output/ for the web site's home page.

    python scripts/make_example_thumbs.py
evolution_output/ is generated locally by a run (python -m evolutionary_art) and is not kept in git; the committed
thumbnails in evolutionary_art/web/static/img/ come from the run that used to be committed there.
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
