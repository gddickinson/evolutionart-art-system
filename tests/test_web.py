"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolutionary_art.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests -------------------------------------------------------------------------------

def test_info_counts_come_from_package():
    from evolutionary_art.styles import ArtisticIntent
    info = client.get("/api/info").json()
    assert info["moods"] == len(ArtisticIntent.MOODS) and info["styles"] == len(ArtisticIntent.STYLES)
    assert info["population_size"] == 20 and info["functions"] == 8


def test_examples_exist():
    for e in client.get("/api/examples").json():
        assert (STATIC / e["src"].removeprefix("static/")).exists()
        assert (ROOT / e["original"]).exists()


@pytest.mark.parametrize("body", [
    {"mode": "basic", "seed": 2, "size": 24, "generations": 2},
    {"mode": "enhanced", "seed": 2, "size": 24, "generations": 2, "style": "geometric", "mood": "calm"},
])
def test_evolve_runs_python_package(body):
    r = client.post("/api/evolve", json=body)
    assert r.status_code == 200, r.text
    gens = r.json()["generations"]
    assert [g["generation"] for g in gens] == [0, 1]
    assert all(g["png"].startswith("data:image/png;base64,") and len(g["scores"]) == 20 for g in gens)
    again = client.post("/api/evolve", json=body).json()["generations"]
    assert [g["scores"] for g in again] == [g["scores"] for g in gens]  # seeded: reproducible


def test_evolve_rejects_bad_input():
    assert client.post("/api/evolve", json={"mode": "other"}).status_code == 422
    assert client.post("/api/evolve", json={"style": "cubist"}).status_code == 422
    assert client.post("/api/evolve", json={"size": 5000}).status_code == 422


def _close(a, b, path="$"):
    """Equal up to floating-point noise: floats within 1e-9 relative or 1e-6 absolute (OpenCV's float32 bilateral
    filter can round differently between builds); integer pixel lists may differ by one level in a few places."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, list) and x and all(isinstance(v, int) and not isinstance(v, bool) for v in x) and len(x) > 50:
                diff = [abs(i - j) for i, j in zip(x, y) if i != j]
                assert len(x) == len(y) and max(diff, default=0) <= 1 and len(diff) <= len(x) * 1e-2, path
            else:
                _close(x, y, f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= max(1e-9 * abs(a), 1e-6), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_js_fixture_is_current():
    """tests/fixtures/evo_reference.json must match a fresh run of the Python code (run scripts/make_js_fixtures.py)."""
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_js_fixtures
    from evolutionary_art.web.reference import fixture
    _close(json.loads(make_js_fixtures.FIXTURE.read_text()), json.loads(make_js_fixtures.dumps(fixture())))
