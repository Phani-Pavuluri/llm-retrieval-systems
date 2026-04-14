#!/usr/bin/env python3
"""Run the Phase 5.2 FastAPI server (development). From repo root: python scripts/run_api.py"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
