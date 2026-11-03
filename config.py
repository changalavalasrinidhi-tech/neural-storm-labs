"""Central configuration for Neural Storm FT-02.

Defines filesystem paths, runtime device selection, MIME routing tables,
default fusion weights, temperature scaling, and uncertainty policy used
by every detector and the Streamlit orchestrator.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger("neural_storm.config")

# ---------------------------------------------------------------------------
# Paths  (/models, /temp relative to the project root; overridable via env)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = Path(os.environ.get("NS_MODELS_DIR", ROOT_DIR / "models"))
TEMP_DIR: Path = Path(os.environ.get("NS_TEMP_DIR", ROOT_DIR / "temp"))

CHECKPOINT_VISUAL: Path = MODELS_DIR / "visual_detector.pt"
CHECKPOINT_AUDIO: Path = MODELS_DIR / "audio_detector.pt"
CHECKPOINT_SYNC: Path = MODELS_DIR / "syncnet.pt"
CHECKPOINT_WAVLM: Path = MODELS_DIR / "wavlm_base.pt"
CHECKPOINT_TEMPERATURE: Path = MODELS_DIR / "temperature_scaler.pt"

# ---------------------------------------------------------------------------
# Device auto-selection
# ---------------------------------------------------------------------------
DEVICE_ENV: str = os.environ.get("NS_DEVICE", "auto").strip().lower()


def resolve_device(preference: Optional[str] = None) -> str:
    """Return ``cuda`` when a GPU is available and requested, otherwise ``cpu``.

    Parameters
    ----------
    preference:
        ``auto``, ``cuda``, or ``cpu``. ``None`` falls back to ``DEVICE_ENV``.
    """
    choice = (preference or DEVICE_ENV or "auto").strip().lower()
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - import-time safety
        logger.warning("PyTorch unavailable during device probe: %s", exc)
        return "cpu"

    if choice in {"cpu", "cpu-only"}:
        return "cpu"
    if choice in {"cuda", "gpu"}:
        if cuda_ok:
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return "cuda" if cuda_ok else "cpu"


DEVICE: str = resolve_device()

# ---------------------------------------------------------------------------
# MIME / extension routing
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
AUDIO_EXTENSIONS: FrozenSet[str] = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
VIDEO_EXTENSIONS: FrozenSet[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})

MIME_TYPE_MAP: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
}

MEDIA_KIND_MAP: Dict[str, str] = {
    **{ext: "image" for ext in IMAGE_EXTENSIONS},
    **{ext: "audio" for ext in AUDIO_EXTENSIONS},
    **{ext: "video" for ext in VIDEO_EXTENSIONS},
}

# ---------------------------------------------------------------------------
# Fusion weights, temperature scaling, uncertainty policy
# ---------------------------------------------------------------------------
DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "w_v": 0.45,
    "w_a": 0.35,
    "w_s": 0.20,
}
W_VISUAL: float = DEFAULT_FUSION_WEIGHTS["w_v"]
W_AUDIO: float = DEFAULT_FUSION_WEIGHTS["w_a"]
W_SYNC: float = DEFAULT_FUSION_WEIGHTS["w_s"]

TEMPERATURE: float = 1.5
UNCERTAIN_SCORE_LOW: float = 0.40
UNCERTAIN_SCORE_HIGH: float = 0.60
EPISTEMIC_UNCERTAINTY_THRESHOLD: float = 0.25

LABEL_AUTHENTIC: str = "Authentic"
LABEL_SYNTHETIC: str = "Synthetic / Deepfake"
LABEL_UNCERTAIN: str = "Uncertain / Requires Forensic Review"

# ---------------------------------------------------------------------------
# Ingestion / detector hyper-parameters
# ---------------------------------------------------------------------------
FACE_SIZE: Tuple[int, int] = (224, 224)
BBOX_PADDING: float = 0.15
TARGET_VIDEO_FPS: int = 30
AUDIO_SAMPLE_RATE: int = 16_000
N_MELS: int = 80
N_FFT: int = 1024
HOP_LENGTH: int = 256
MAX_AUDIO_SECONDS: float = 12.0
MAX_FRAMES_CPU: int = 16
MAX_FRAMES_CUDA: int = 48
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

SUPPORTED_UPLOAD_TYPES: Tuple[str, ...] = (
    "mp4",
    "avi",
    "mov",
    "jpg",
    "jpeg",
    "png",
    "wav",
    "mp3",
)


def ensure_runtime_dirs() -> None:
    """Create ``/models`` and ``/temp`` if they do not already exist."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Unable to create runtime directories: %s", exc)
        raise


ensure_runtime_dirs()
