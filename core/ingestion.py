"""Universal media ingestion and routing for Neural Storm FT-02.

Inspects MIME / extension, demuxes video with FFmpeg, crops faces with
MediaPipe Face Mesh (OpenCV Haar fallback), and builds 16 kHz log-mel
spectrograms for the audio stream.
"""

from __future__ import annotations

import logging
import mimetypes
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

logger = logging.getLogger("neural_storm.ingestion")

# MediaPipe Face Mesh lip landmark indices (outer + inner mouth).
_LIP_LANDMARKS: Tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191,
)


@dataclass
class IngestedMedia:
    """Normalized payload produced by :class:`MediaIngestor`."""

    media_type: str
    source_path: Path
    mime_type: str
    faces: List[np.ndarray] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)
    face_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    lip_rois: List[np.ndarray] = field(default_factory=list)
    audio_waveform: Optional[np.ndarray] = None
    sample_rate: int = config.AUDIO_SAMPLE_RATE
    log_mel: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0

    @property
    def has_audio(self) -> bool:
        return self.audio_waveform is not None and self.audio_waveform.size > 0

    @property
    def has_speech_and_faces(self) -> bool:
        return self.has_faces and self.has_audio


class MediaIngestor:
    """Universal media router for images, audio, and video."""

    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        face_size: Tuple[int, int] = config.FACE_SIZE,
        bbox_padding: float = config.BBOX_PADDING,
        target_fps: int = config.TARGET_VIDEO_FPS,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        max_frames: Optional[int] = None,
    ) -> None:
        self.temp_dir = Path(temp_dir or config.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.face_size = face_size
        self.bbox_padding = bbox_padding
        self.target_fps = int(target_fps)
        self.sample_rate = int(sample_rate)
        self.max_frames = max_frames
        self._face_mesh = self._init_face_mesh()
        self._haar = self._init_haar()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(self, path: str | Path) -> IngestedMedia:
        """Inspect ``path`` and dispatch to the matching modality pipeline."""
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Media file not found: {src}")

        mime_type, media_type = self.inspect_format(src)
        logger.info("Ingesting %s as %s (%s)", src.name, media_type, mime_type)

        if media_type == "image":
            return self._ingest_image(src, mime_type)
        if media_type == "audio":
            return self._ingest_audio(src, mime_type)
        if media_type == "video":
            return self._ingest_video(src, mime_type)
        raise ValueError(f"Unsupported media type for {src}: {mime_type}")

    def inspect_format(self, path: Path) -> Tuple[str, str]:
        """Return ``(mime_type, media_kind)`` for ``path``."""
        ext = path.suffix.lower()
        mime = config.MIME_TYPE_MAP.get(ext)
        if mime is None:
            guessed, _ = mimetypes.guess_type(str(path))
            mime = guessed or "application/octet-stream"
        kind = config.MEDIA_KIND_MAP.get(ext)
        if kind is None:
            if mime.startswith("image/"):
                kind = "image"
            elif mime.startswith("audio/"):
                kind = "audio"
            elif mime.startswith("video/"):
                kind = "video"
            else:
                raise ValueError(f"Unrecognized format: {path.suffix} ({mime})")
        return mime, kind

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def _ingest_image(self, src: Path, mime_type: str) -> IngestedMedia:
        frame = self._read_bgr(src)
        rgb = self._bgr_to_rgb(frame)
        faces, boxes, lips = self._extract_faces([rgb])
        return IngestedMedia(
            media_type="image",
            source_path=src,
            mime_type=mime_type,
            faces=faces or [self._center_crop_resize(rgb)],
            frames=[rgb],
            face_boxes=boxes,
            lip_rois=lips,
            metadata={"height": rgb.shape[0], "width": rgb.shape[1], "n_faces": len(faces)},
        )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def _ingest_audio(self, src: Path, mime_type: str) -> IngestedMedia:
        waveform, sr = self._load_audio(src)
        log_mel = self.compute_log_mel(waveform, sr)
        return IngestedMedia(
            media_type="audio",
            source_path=src,
            mime_type=mime_type,
            audio_waveform=waveform,
            sample_rate=sr,
            log_mel=log_mel,
            metadata={"duration_sec": float(waveform.size / max(sr, 1)), "n_mels": log_mel.shape[0]},
        )

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    def _ingest_video(self, src: Path, mime_type: str) -> IngestedMedia:
        job_dir = self.temp_dir / f"job_{uuid.uuid4().hex}"
        job_dir.mkdir(parents=True, exist_ok=True)
        wav_path = job_dir / "audio_16k_mono.wav"
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        frames = self._extract_video_frames(src, frames_dir)
        waveform, sr = self._demux_audio(src, wav_path)
        log_mel = self.compute_log_mel(waveform, sr) if waveform.size else None
        faces, boxes, lips = self._extract_faces(frames)
        if not faces and frames:
            faces = [self._center_crop_resize(frames[0])]

        return IngestedMedia(
            media_type="video",
            source_path=src,
            mime_type=mime_type,
            faces=faces,
            frames=frames,
            face_boxes=boxes,
            lip_rois=lips,
            audio_waveform=waveform if waveform.size else None,
            sample_rate=sr,
            log_mel=log_mel,
            metadata={
                "n_frames": len(frames),
                "n_faces": len(faces),
                "has_audio": bool(waveform.size),
                "target_fps": self.target_fps,
                "job_dir": str(job_dir),
            },
        )

    MAX_FRAME_SIDE = 1280  # downscale big screens/4K so memory stays bounded

    def _frame_cap(self) -> int:
        if self.max_frames is not None:
            return int(self.max_frames)
        return config.MAX_FRAMES_CUDA if config.resolve_device() == "cuda" else config.MAX_FRAMES_CPU

    def _shrink(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        side = max(h, w)
        if side <= self.MAX_FRAME_SIDE:
            return rgb
        s = self.MAX_FRAME_SIDE / float(side)
        return self._resize_rgb(rgb, (max(1, int(w * s)), max(1, int(h * s))))

    def _extract_video_frames(self, src: Path, frames_dir: Path) -> List[np.ndarray]:
        """Sample at most ``max_frames`` frames spread across the WHOLE video.

        The previous implementation decoded every frame at 30 FPS to PNG (a 2.5-minute clip meant
        ~4,700 files and tens of GB of RAM) before discarding all but a handful. Now only the
        frames that are actually analysed are ever decoded.
        """
        frames = self._sample_frames_opencv(src)
        if frames:
            return frames
        return self._sample_frames_ffmpeg(src, frames_dir)

    def _sample_frames_opencv(self, src: Path) -> List[np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            logger.warning("OpenCV could not open video %s", src)
            return []
        want = max(1, self._frame_cap())
        frames: List[np.ndarray] = []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                for idx in np.linspace(0, max(total - 1, 0), num=min(want, total), dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, bgr = cap.read()
                    if ok and bgr is not None:
                        frames.append(self._shrink(self._bgr_to_rgb(bgr)))
                if len(frames) >= max(1, min(want, total) // 2):
                    return frames
                logger.warning("Seek-sampling returned %d/%d frames; scanning sequentially.", len(frames), want)
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Unknown length or broken seeking: bounded sequential scan, keep an even subset.
            kept: List[np.ndarray] = []
            stride, seen = 1, 0
            while seen < 20000:
                ok, bgr = cap.read()
                if not ok:
                    break
                if seen % stride == 0:
                    kept.append(self._shrink(self._bgr_to_rgb(bgr)))
                    if len(kept) >= want * 2:  # thin out and double the stride
                        kept, stride = kept[::2], stride * 2
                seen += 1
            return self._limit_frames(kept)
        finally:
            cap.release()

    def _sample_frames_ffmpeg(self, src: Path, frames_dir: Path) -> List[np.ndarray]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return []
        want = max(1, self._frame_cap())
        duration = 0.0
        try:
            import cv2

            cap = cv2.VideoCapture(str(src))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap.release()
            duration = n / fps if fps > 0 and n > 0 else 0.0
        except Exception:
            pass
        rate = (want / duration) if duration > 0 else 1.0
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-vf", f"fps={rate:.6f},scale='min({self.MAX_FRAME_SIDE},iw)':-2",
            "-frames:v", str(want), str(frames_dir / "frame_%06d.png"),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=240)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("FFmpeg frame sampling failed: %s", exc)
        files = sorted(frames_dir.glob("frame_*.png"))[:want]
        return [self._bgr_to_rgb(self._read_bgr(p)) for p in files]

    def _demux_audio(self, src: Path, wav_path: Path) -> Tuple[np.ndarray, int]:
        """Extract 16 kHz mono PCM via FFmpeg, else torchaudio/librosa."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-t",
                str(config.MAX_AUDIO_SECONDS),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                str(wav_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=180)
                if wav_path.is_file() and wav_path.stat().st_size > 0:
                    return self._load_audio(wav_path)
            except (subprocess.SubprocessError, OSError, TimeoutError) as exc:
                logger.warning("FFmpeg audio demux failed (%s); trying loaders.", exc)
        try:
            return self._load_audio(src)
        except Exception as exc:
            logger.warning("No audio track recovered from %s: %s", src.name, exc)
            return np.zeros((0,), dtype=np.float32), self.sample_rate

    # ------------------------------------------------------------------
    # Face crop
    # ------------------------------------------------------------------
    def _extract_faces(
        self, frames: Sequence[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[np.ndarray]]:
        faces: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []
        lips: List[np.ndarray] = []
        for rgb in frames:
            box, lip = self._detect_face(rgb)
            if box is None:
                continue
            crop = self._crop_padded(rgb, box)
            faces.append(crop)
            boxes.append(box)
            if lip is not None:
                lips.append(lip)
            else:
                lips.append(self._lip_from_box(rgb, box))
        return faces, boxes, lips

    def _detect_face(
        self, rgb: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        h, w = rgb.shape[:2]
        if self._face_mesh is not None:
            try:
                result = self._face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    xs = [lm.x * w for lm in lms]
                    ys = [lm.y * h for lm in lms]
                    box = self._clamp_box(
                        int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), w, h
                    )
                    lip_pts = np.array(
                        [[lms[i].x * w, lms[i].y * h] for i in _LIP_LANDMARKS if i < len(lms)],
                        dtype=np.float32,
                    )
                    lip = self._crop_lip(rgb, lip_pts) if lip_pts.size else None
                    return box, lip
            except Exception as exc:
                logger.debug("MediaPipe Face Mesh failed: %s", exc)

        if self._haar is not None:
            try:
                import cv2

                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                detections = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
                if len(detections) > 0:
                    x, y, bw, bh = max(detections, key=lambda d: d[2] * d[3])
                    box = self._clamp_box(int(x), int(y), int(x + bw), int(y + bh), w, h)
                    return box, self._lip_from_box(rgb, box)
            except Exception as exc:
                logger.debug("Haar face detector failed: %s", exc)
        return None, None

    def _crop_padded(self, rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        pad_x = int(bw * self.bbox_padding)
        pad_y = int(bh * self.bbox_padding)
        h, w = rgb.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = rgb[y1:y2, x1:x2]
        return self._resize_rgb(crop, self.face_size)

    def _crop_lip(self, rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        box = self._clamp_box(int(x1), int(y1), int(x2), int(y2), rgb.shape[1], rgb.shape[0])
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)
        pad = int(0.25 * max(bw, bh))
        h, w = rgb.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return self._resize_rgb(rgb[y1:y2, x1:x2], (96, 96))

    def _lip_from_box(self, rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        lower = y1 + int(0.62 * (y2 - y1))
        return self._resize_rgb(rgb[lower:y2, x1:x2], (96, 96))

    # ------------------------------------------------------------------
    # Audio DSP
    # ------------------------------------------------------------------
    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        waveform, sr = self._load_audio_backend(path)
        waveform = self._to_mono(waveform)
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate
        max_len = int(config.MAX_AUDIO_SECONDS * sr)
        if waveform.size > max_len:
            waveform = waveform[:max_len]
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1e-6:
            waveform = (waveform / peak).astype(np.float32)
        else:
            waveform = waveform.astype(np.float32)
        return waveform, sr

    def _load_audio_backend(self, path: Path) -> Tuple[np.ndarray, int]:
        try:
            import torchaudio

            wav, sr = torchaudio.load(str(path))
            return wav.mean(dim=0).numpy().astype(np.float32), int(sr)
        except Exception as exc:
            logger.debug("torchaudio.load failed (%s); trying librosa.", exc)
        try:
            import librosa

            wav, sr = librosa.load(str(path), sr=None, mono=True)
            return np.asarray(wav, dtype=np.float32), int(sr)
        except Exception as exc:
            raise RuntimeError(f"Unable to decode audio from {path}") from exc

    def compute_log_mel(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute an STFT log-mel spectrogram (n_mels, time)."""
        if waveform.size == 0:
            return np.zeros((config.N_MELS, 1), dtype=np.float32)
        try:
            import torch
            import torchaudio

            wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS,
                power=2.0,
            )(wav)
            log_mel = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
            return log_mel.squeeze(0).numpy().astype(np.float32)
        except Exception:
            import librosa

            spec = librosa.feature.melspectrogram(
                y=np.asarray(waveform, dtype=np.float32),
                sr=sample_rate,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS,
                power=2.0,
            )
            return librosa.power_to_db(spec, ref=np.max).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _limit_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        cap = self.max_frames
        if cap is None:
            cap = config.MAX_FRAMES_CUDA if config.resolve_device() == "cuda" else config.MAX_FRAMES_CPU
        if len(frames) <= cap:
            return frames
        idx = np.linspace(0, len(frames) - 1, num=cap, dtype=int)
        return [frames[i] for i in idx]

    def _init_face_mesh(self) -> Any:
        try:
            import mediapipe as mp

            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
            )
        except Exception as exc:
            logger.warning("MediaPipe Face Mesh unavailable (%s); Haar/center-crop fallback.", exc)
            return None

    def _init_haar(self) -> Any:
        try:
            import cv2

            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if cascade.empty():
                return None
            return cascade
        except Exception as exc:
            logger.warning("OpenCV Haar cascade unavailable: %s", exc)
            return None

    @staticmethod
    def _read_bgr(path: Path) -> np.ndarray:
        try:
            import cv2

            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            pass
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))[:, :, ::-1]

    @staticmethod
    def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            return np.stack([bgr, bgr, bgr], axis=-1)
        try:
            import cv2

            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return bgr[:, :, ::-1].copy()

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if rgb.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        try:
            import cv2

            return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))

    def _center_crop_resize(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return self._resize_rgb(rgb[y0 : y0 + side, x0 : x0 + side], self.face_size)

    @staticmethod
    def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=0 if arr.shape[0] <= 8 else 1)

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import torch
            import torchaudio

            wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            resampled = torchaudio.functional.resample(wav, orig_sr, target_sr)
            return resampled.squeeze(0).numpy()
        except Exception:
            import librosa

            return librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr)

    def close(self) -> None:
        mesh = self._face_mesh
        if mesh is not None:
            try:
                mesh.close()
            except Exception:
                pass

    def __enter__(self) -> "MediaIngestor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
