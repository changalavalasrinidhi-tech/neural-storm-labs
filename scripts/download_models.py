"""Download the MediaPipe Tasks model files used by newer mediapipe versions.

Usage:  python scripts/download_models.py
Only needed when `mediapipe` > 0.10.21 (no `mp.solutions`). The pinned version in
requirements.txt does not need these files.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.landmark_backend import download_models  # noqa: E402

if __name__ == "__main__":
    for line in download_models(ROOT / "models"):
        print(line)
