"""Project file explorer and environment health check (torch-free)."""

from __future__ import annotations

import importlib
import importlib.util
import re
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache"}
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".json", ".csv", ".yml", ".yaml", ".cfg", ".ini", ".html", ".gitkeep", ""}
DESCRIPTIONS = {
    "app.py": "Streamlit UI and pipeline orchestrator (tabs: Studio, Camera/Mic/Hand Lab, Tamper Compare, Explorer, Research, Session log & custody).",
    "config.py": "Central paths, device selection, MIME routing, fusion weights, thresholds.",
    "requirements.txt": "Python dependencies.",
    "ARCHITECTURE.md": "Original architecture specification.",
    "REDESIGN_NOTES.md": "Notes from the UI redesign.",
    "CHANGELOG.md": "What changed in v2 and what is verified.",
    "README.md": "How to install, run and test.",
    "ui_labs.py": "Streamlit rendering for the Labs tabs (image metrics, hands, liveness, explorer, research, log).",
    "core/ingestion.py": "Universal media validation, FFmpeg demux, face crop (MediaPipe / Haar).",
    "core/visual_detector.py": "EfficientNet-B4 + DCT/SRM dual-domain visual detector.",
    "core/audio_detector.py": "WavLM/AASIST-style voice spoof detector.",
    "core/sync_engine.py": "SyncNet-style lip-sync consistency engine.",
    "core/fusion.py": "Quality-weighted logit fusion and temperature calibration.",
    "core/explainability.py": "Grad-CAM++ heatmaps and spectrogram highlighting.",
    "labs/image_metrics.py": "Image forensics: quality, noise, FFT, ELA, EXIF, provenance, face-vs-scene.",
    "labs/hand_analytics.py": "Hand joint angles, proportions, gesture classifier, plausibility checks.",
    "labs/landmark_backend.py": "Face/hand detection across MediaPipe generations + OpenCV fallback + drawing.",
    "labs/liveness.py": "Experimental gesture challenge-response.",
    "labs/report.py": "JSON and self-contained HTML evidence reports; session log rows.",
    "labs/explorer.py": "This file explorer and health check.",
    "labs/input_profile.py": "Input dossier (what was submitted) + torch-free audio forensics lab for audio files, video audio tracks and mic clips.",
    "labs/triage.py": "Transparent 0-100 review-priority score with per-signal breakdown and coverage.",
    "labs/custody.py": "Tamper-evident SHA-256 hash chain for the session log.",
    "labs/compare.py": "Original-vs-suspect image comparison (SSIM, diff heat-map, changed regions, hashes).",
    "labs/demo.py": "Built-in synthetic demo voice clip and upload stand-in.",
    "labs/research_content.py": "Research notes, references and metric glossary.",
    "labs/synthetic.py": "Synthetic 3-D hands for tests and demo.",
    "scripts/download_models.py": "Downloads MediaPipe Tasks models for newer mediapipe versions.",
}


def scan_project(root: Path) -> List[Dict[str, Any]]:
    root = Path(root).resolve()
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS or part.startswith(("job_", "ns_upload_")) for part in rel.parts) or p.suffix == ".pyc" or p.is_dir():
            continue
        text = p.suffix.lower() in TEXT_EXT
        lines: Optional[int] = None
        if text:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except OSError:
                lines = None
        rows.append({"path": rel.as_posix(), "type": p.suffix.lower() or "(none)", "size_bytes": p.stat().st_size,
                     "lines": lines, "text": text, "description": DESCRIPTIONS.get(rel.as_posix(), "")})
    return rows


def read_text(root: Path, rel: str, max_bytes: int = 250_000) -> str:
    """Read a project text file safely (no path traversal)."""
    root = Path(root).resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError("path escapes the project directory")
    data = target.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _parse_requirements(path: Path) -> List[Dict[str, str]]:
    reqs: List[Dict[str, str]] = []
    if not path.is_file():
        return reqs
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if m:
            reqs.append({"name": m.group(1), "spec": m.group(2).strip()})
    return reqs


def _satisfied(version: str, spec: str) -> Optional[bool]:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(version) in SpecifierSet(spec)
    except Exception:
        return None


def health_check(root: Path) -> List[Dict[str, str]]:
    root = Path(root)
    rows: List[Dict[str, str]] = []

    def add(item: str, status: str, detail: str) -> None:
        rows.append({"item": item, "status": status, "detail": detail})

    for req in _parse_requirements(root / "requirements.txt"):
        try:
            v = metadata.version(req["name"])
            ok = _satisfied(v, req["spec"])
            add(f"package: {req['name']}", "ok" if ok else "warn" if ok is False else "ok",
                f"installed {v}" + (f" - requirement '{req['spec']}' NOT satisfied" if ok is False else f" (wants {req['spec'] or 'any'})"))
        except metadata.PackageNotFoundError:
            add(f"package: {req['name']}", "missing", f"not installed (wants {req['spec'] or 'any'})")

    add("ffmpeg", "ok" if shutil.which("ffmpeg") else "warn",
        shutil.which("ffmpeg") or "not on PATH - video falls back to OpenCV frames / no audio demux")

    try:
        import mediapipe as mp  # type: ignore

        has_sol = hasattr(mp, "solutions")
        add("mediapipe API", "ok" if has_sol else "warn",
            f"{getattr(mp, '__version__', '?')}: " + ("legacy mp.solutions available" if has_sol else "no mp.solutions - Tasks API + model files required (scripts/download_models.py)"))
    except Exception as exc:
        add("mediapipe API", "missing", f"cannot import: {exc}")

    for fn, label in (("hand_landmarker.task", "hand model (Tasks API)"), ("blaze_face_short_range.tflite", "face model (Tasks API)")):
        p = root / "models" / fn
        add(f"models/{fn}", "ok" if p.is_file() else "info", f"{label}: {'present' if p.is_file() else 'absent (only needed for mediapipe > 0.10.21)'}")
    for fn in ("visual_detector.pt", "audio_detector.pt", "syncnet.pt", "temperature_scaler.pt"):
        p = root / "models" / fn
        add(f"models/{fn}", "ok" if p.is_file() else "info", "trained checkpoint present" if p.is_file() else "absent - detector runs in heuristic fallback mode")

    for mod in ("labs.image_metrics", "labs.hand_analytics", "labs.landmark_backend", "labs.liveness", "labs.report", "labs.explorer",
                "labs.input_profile", "labs.triage", "labs.custody", "labs.compare", "core"):
        try:
            importlib.import_module(mod)
            add(f"import {mod}", "ok", "imports cleanly")
        except Exception as exc:
            add(f"import {mod}", "missing", f"{type(exc).__name__}: {exc}")

    try:
        import torch  # type: ignore

        add("CUDA", "ok" if torch.cuda.is_available() else "info", "GPU visible" if torch.cuda.is_available() else "CPU only")
    except Exception:
        add("CUDA", "info", "PyTorch not installed")
    return rows
