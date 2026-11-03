"""Face + hand detection that works across MediaPipe generations.

* ``solutions``  - legacy ``mp.solutions`` API (mediapipe <= 0.10.21, the pinned version).
* ``tasks``      - ``mp.tasks.vision`` API (newer mediapipe, where ``mp.solutions`` was removed).
                   Needs two model files in ``models/`` (see ``scripts/download_models.py``).
* ``opencv-haar``- face boxes only (always available when OpenCV is). No hand detection.

The backend never raises on missing models/packages: it reports what is unavailable in ``notes``.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from labs.hand_analytics import FINGERS, FINGER_ORDER, HAND_CONNECTIONS

logger = logging.getLogger("neural_storm.labs.landmarks")

HAND_MODEL_NAME = "hand_landmarker.task"
FACE_MODEL_NAME = "blaze_face_short_range.tflite"
MODEL_URLS = {
    HAND_MODEL_NAME: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    FACE_MODEL_NAME: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}
FINGER_COLORS = {  # RGB
    "thumb": (255, 159, 67), "index": (255, 221, 89), "middle": (97, 242, 194),
    "ring": (86, 204, 242), "pinky": (233, 120, 255), "palm": (230, 236, 245),
}


@dataclass
class FaceDet:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    score: Optional[float]          # None when the detector gives no probability (Haar)
    source: str


@dataclass
class HandDet:
    landmarks: np.ndarray           # (21, 3) normalised
    label: str                      # "Left" / "Right" as reported by the model (image may be mirrored)
    score: Optional[float]


@dataclass
class DetectionResult:
    faces: List[FaceDet] = field(default_factory=list)
    hands: List[HandDet] = field(default_factory=list)
    backend: str = "none"
    hands_available: bool = False
    faces_available: bool = False
    notes: List[str] = field(default_factory=list)


def download_models(models_dir: Path, timeout: float = 30.0) -> List[str]:
    """Download the two Tasks-API model files. Returns human-readable status lines."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    msgs: List[str] = []
    for name, url in MODEL_URLS.items():
        dest = models_dir / name
        if dest.is_file() and dest.stat().st_size > 1000:
            msgs.append(f"{name}: already present")
            continue
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:  # noqa: S310
                fh.write(resp.read())
            tmp.replace(dest)
            msgs.append(f"{name}: downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            msgs.append(f"{name}: download failed ({exc}). Download it manually from {url}")
    return msgs


class LandmarkBackend:
    def __init__(self, models_dir: Path, min_confidence: float = 0.5, max_hands: int = 4) -> None:
        self.models_dir = Path(models_dir)
        self.min_conf = float(min_confidence)
        self.max_hands = int(max_hands)
        self._mp: Any = None
        self.mode = "opencv-haar"
        self.mp_version: Optional[str] = None
        self._tasks_hands: Any = None
        self._tasks_faces: Any = None
        self._haar = self._init_haar()
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self.mp_version = getattr(mp, "__version__", "unknown")
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.mode = "solutions"
            elif hasattr(mp, "tasks"):
                self.mode = "tasks"
        except Exception as exc:
            logger.info("MediaPipe not importable: %s", exc)
            self.import_error = str(exc)

    # ------------------------------------------------------------------ status
    @property
    def hand_model_path(self) -> Path:
        return self.models_dir / HAND_MODEL_NAME

    @property
    def face_model_path(self) -> Path:
        return self.models_dir / FACE_MODEL_NAME

    def status(self) -> Dict[str, Any]:
        tasks_ready = self.mode == "tasks" and self.hand_model_path.is_file()
        return {
            "mode": self.mode,
            "mediapipe_version": self.mp_version,
            "hands_ready": self.mode == "solutions" or tasks_ready,
            "face_model": "mediapipe" if (self.mode == "solutions" or (self.mode == "tasks" and self.face_model_path.is_file())) else "opencv-haar" if self._haar is not None else "none",
            "needs_model_download": self.mode == "tasks" and not (self.hand_model_path.is_file() and self.face_model_path.is_file()),
            "models_dir": str(self.models_dir),
        }

    # ------------------------------------------------------------------ public
    def detect(self, rgb: np.ndarray) -> DetectionResult:
        rgb = np.ascontiguousarray(self._to_uint8_rgb(rgb))
        res = DetectionResult(backend=self.mode)
        h, w = rgb.shape[:2]

        if self.mode == "solutions":
            self._detect_solutions(rgb, res, w, h)
        elif self.mode == "tasks":
            self._detect_tasks(rgb, res, w, h)
        else:
            res.notes.append("MediaPipe is not installed: hand detection is unavailable (faces use OpenCV Haar).")

        if not res.faces_available:
            self._detect_haar(rgb, res)
        return res

    # ------------------------------------------------------------------ solutions API
    def _detect_solutions(self, rgb: np.ndarray, res: DetectionResult, w: int, h: int) -> None:
        mp = self._mp
        try:
            with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=self.min_conf) as fd:
                out = fd.process(rgb)
            res.faces_available = True
            for det in out.detections or []:
                b = det.location_data.relative_bounding_box
                x1, y1 = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
                x2, y2 = min(w - 1, int((b.xmin + b.width) * w)), min(h - 1, int((b.ymin + b.height) * h))
                res.faces.append(FaceDet((x1, y1, x2, y2), float(det.score[0]) if det.score else None, "mediapipe"))
        except Exception as exc:
            res.notes.append(f"MediaPipe face detector failed ({exc}); using OpenCV Haar.")
        try:
            with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=self.max_hands,
                                          min_detection_confidence=self.min_conf) as hands:
                out = hands.process(rgb)
            res.hands_available = True
            for i, hl in enumerate(out.multi_hand_landmarks or []):
                lm = np.array([[p.x, p.y, p.z] for p in hl.landmark], dtype=np.float64)
                label, score = "Unknown", None
                if out.multi_handedness and i < len(out.multi_handedness):
                    c = out.multi_handedness[i].classification[0]
                    label, score = c.label, float(c.score)
                res.hands.append(HandDet(lm, label, score))
        except Exception as exc:
            res.notes.append(f"MediaPipe hand detector failed: {exc}")

    # ------------------------------------------------------------------ tasks API
    def _detect_tasks(self, rgb: np.ndarray, res: DetectionResult, w: int, h: int) -> None:
        mp = self._mp
        if not self.hand_model_path.is_file():
            res.notes.append(
                f"This MediaPipe ({self.mp_version}) uses the Tasks API and needs '{HAND_MODEL_NAME}' in {self.models_dir}. "
                "Run `python scripts/download_models.py` (or press 'Download models' in the app), "
                "or install the pinned `mediapipe==0.10.21`."
            )
        else:
            try:
                if self._tasks_hands is None:
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision

                    opts = vision.HandLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(self.hand_model_path)),
                        running_mode=vision.RunningMode.IMAGE, num_hands=self.max_hands,
                        min_hand_detection_confidence=self.min_conf, min_hand_presence_confidence=self.min_conf,
                    )
                    self._tasks_hands = vision.HandLandmarker.create_from_options(opts)
                out = self._tasks_hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                res.hands_available = True
                for i, pts in enumerate(out.hand_landmarks or []):
                    lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
                    label, score = "Unknown", None
                    if out.handedness and i < len(out.handedness) and out.handedness[i]:
                        label, score = out.handedness[i][0].category_name, float(out.handedness[i][0].score)
                    res.hands.append(HandDet(lm, label, score))
            except Exception as exc:
                res.notes.append(f"MediaPipe Tasks hand detector failed: {exc}")

        if self.face_model_path.is_file():
            try:
                if self._tasks_faces is None:
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision

                    opts = vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(self.face_model_path)),
                        running_mode=vision.RunningMode.IMAGE, min_detection_confidence=self.min_conf,
                    )
                    self._tasks_faces = vision.FaceDetector.create_from_options(opts)
                out = self._tasks_faces.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                res.faces_available = True
                for d in out.detections or []:
                    bb = d.bounding_box
                    x1, y1 = max(0, bb.origin_x), max(0, bb.origin_y)
                    x2, y2 = min(w - 1, bb.origin_x + bb.width), min(h - 1, bb.origin_y + bb.height)
                    res.faces.append(FaceDet((x1, y1, x2, y2), float(d.categories[0].score) if d.categories else None, "mediapipe"))
            except Exception as exc:
                res.notes.append(f"MediaPipe Tasks face detector failed ({exc}); using OpenCV Haar.")

    # ------------------------------------------------------------------ Haar fallback
    def _init_haar(self) -> Any:
        try:
            casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            return None if casc.empty() else casc
        except Exception:
            return None

    def _detect_haar(self, rgb: np.ndarray, res: DetectionResult) -> None:
        if self._haar is None:
            res.notes.append("No face detector available (OpenCV Haar cascade missing).")
            return
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        min_side = max(48, int(min(gray.shape[:2]) * 0.08))
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        res.faces = [FaceDet((int(x), int(y), int(x + bw), int(y + bh)), None, "opencv-haar") for x, y, bw, bh in boxes]
        res.faces_available = True
        if self.mode != "solutions":
            res.notes.append("Faces found with OpenCV Haar (frontal faces only, no confidence score).")

    @staticmethod
    def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
        return a

    def close(self) -> None:
        for obj in (self._tasks_hands, self._tasks_faces):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------- drawing
def draw_detections(rgb: np.ndarray, res: DetectionResult, show_labels: bool = True) -> np.ndarray:
    """Draw face boxes and coloured hand skeletons on a copy of ``rgb`` (own drawing; no mp.drawing_utils)."""
    img = np.ascontiguousarray(LandmarkBackend._to_uint8_rgb(rgb)).copy()
    h, w = img.shape[:2]
    th = max(2, int(round(min(h, w) / 320)))
    fs = max(0.5, min(h, w) / 900)
    for f in res.faces:
        x1, y1, x2, y2 = f.box
        cv2.rectangle(img, (x1, y1), (x2, y2), (97, 242, 194), th)
        if show_labels:
            txt = "FACE" if f.score is None else f"FACE {f.score:.2f}"
            cv2.putText(img, txt, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, fs, (97, 242, 194), th, cv2.LINE_AA)
    finger_of: Dict[Tuple[int, int], str] = {}
    for name, idx in FINGERS.items():
        for a, b in zip(idx[:-1], idx[1:]):
            finger_of[(a, b)] = name
        finger_of[(0 if name in ("thumb", "pinky") else idx[0] - 1, idx[0])] = name
    for hd in res.hands:
        pts = np.stack([hd.landmarks[:, 0] * w, hd.landmarks[:, 1] * h], axis=1).astype(int)
        for a, b in HAND_CONNECTIONS:
            col = FINGER_COLORS.get(finger_of.get((a, b), "palm"), FINGER_COLORS["palm"])
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, th, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
        if show_labels:
            txt = hd.label if hd.score is None else f"{hd.label} {hd.score:.2f}"
            cv2.putText(img, txt, (int(pts[0][0]), min(h - 6, int(pts[0][1]) + int(26 * fs) + 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return img
