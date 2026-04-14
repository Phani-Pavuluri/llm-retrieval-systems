#!/usr/bin/env python3
"""Launch the Phase 5.3 Streamlit chat UI. Requires the API (Phase 5.2) running separately."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":
    script = _ROOT / "ui" / "chat_ui.py"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
        ]
    )
