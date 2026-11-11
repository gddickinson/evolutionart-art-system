"""Start the Evolutionary Art System web app on http://127.0.0.1:8158 (local only).

    pip install -r requirements-web.txt
    python run_web.py [--port 8158]
"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8158)
    a = ap.parse_args()
    print(f"Evolutionary Art System web app: http://127.0.0.1:{a.port}")
    uvicorn.run("evolutionary_art.web.app:app", host="127.0.0.1", port=a.port)
