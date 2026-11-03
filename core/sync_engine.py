"""SyncNet-style phoneme–viseme alignment engine.

Measures cosine distance between spoken-audio phoneme embeddings and
facial lip-motion vectors. The stream is active only when both a face
track and a speech waveform are present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger("neural_storm.sync")


class VisualLipEncoder(nn.Module):
    """2-D CNN over stacked grayscale lip ROIs (viseme motion)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, lips: torch.Tensor) -> torch.Tensor:
        """lips: ``(B, T, 1, H, W)`` → ``(B, T, D)``."""
        b, t, c, h, w = lips.shape
        feat = self.net(lips.reshape(b * t, c, h, w)).flatten(1)
        return self.proj(feat).view(b, t, -1)


class AudioPhonemeEncoder(nn.Module):
    """2-D CNN over log-mel patches (phoneme proxy embeddings)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: ``(B, 1, n_mels, time)`` → ``(B, T_windows, D)``."""
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        b, _, n_mels, n_t = mel.shape
        win = min(16, max(4, n_t // 8 or 4))
        hop = max(1, win // 2)
        patches = []
        for start in range(0, max(n_t - win, 0) + 1, hop):
            patches.append(mel[:, :, :, start : start + win])
        if not patches:
            patches = [F.pad(mel, (0, max(0, win - n_t)))]
        stacked = torch.stack(patches, dim=1)  # B, P, 1, M, W
        b, p, c, h, w = stacked.shape
        feat = self.net(stacked.reshape(b * p, c, h, w)).flatten(1)
        return self.proj(feat).view(b, p, -1)


class LipSyncEngine(nn.Module):
    """SyncNet phoneme–viseme cosine-distance engine."""

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or config.resolve_device())
        self.checkpoint_path = Path(checkpoint or config.CHECKPOINT_SYNC)
        self.using_fallback_weights = True
        self.visual = VisualLipEncoder(embed_dim=128)
        self.audio = AudioPhonemeEncoder(embed_dim=128)
        self.mismatch_head = nn.Linear(1, 1)
        self._load_or_heuristic_init()
        self.to(self.device)
        self.eval()

    def _load_or_heuristic_init(self) -> None:
        if self.checkpoint_path.is_file():
            try:
                blob = torch.load(self.checkpoint_path, map_location="cpu")
                state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
                self.load_state_dict(state, strict=False)
                self.using_fallback_weights = False
                logger.info("Loaded SyncNet checkpoint %s", self.checkpoint_path.name)
                return
            except Exception as exc:
                logger.warning("SyncNet checkpoint failed (%s); heuristic init.", exc)
        nn.init.constant_(self.mismatch_head.weight, 2.0)
        nn.init.constant_(self.mismatch_head.bias, -0.4)
        logger.warning("No SyncNet weights at %s — cosine-distance fallback.", self.checkpoint_path)

    @torch.no_grad()
    def infer(
        self,
        lip_rois: Sequence[np.ndarray],
        waveform: Optional[np.ndarray],
        log_mel: Optional[np.ndarray],
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        has_faces: bool = True,
    ) -> Dict[str, Any]:
        """Compute lip-sync distance and mismatch logit.

        Inactive unless both facial tracks and a speech stream exist.
        """
        inactive = {
            "active": False,
            "lip_sync_distance": 0.0,
            "mismatch_logit": 0.0,
            "cosine_similarity": 0.0,
            "using_fallback_weights": self.using_fallback_weights,
        }
        if not has_faces or not lip_rois or waveform is None or np.asarray(waveform).size == 0:
            return inactive

        lips = self._prep_lips(lip_rois).to(self.device)
        mel = self._prep_mel(log_mel, waveform, sample_rate).to(self.device)
        viseme = F.normalize(self.visual(lips).mean(dim=1), dim=-1)
        phoneme = F.normalize(self.audio(mel).mean(dim=1), dim=-1)
        cosine = float((viseme * phoneme).sum(dim=-1).clamp(-1.0, 1.0).mean().item())
        distance = float(1.0 - cosine)

        motion = _lip_motion_energy(lip_rois)
        audio_env = _audio_envelope_energy(np.asarray(waveform), sample_rate)
        # Heuristic alignment of motion vs. speech energy when weights are random.
        env_mismatch = abs(motion - audio_env)
        if self.using_fallback_weights:
            distance = 0.55 * distance + 0.45 * float(np.clip(env_mismatch, 0.0, 1.5))
            cosine = 1.0 - distance

        dist_t = torch.tensor([[distance]], device=self.device, dtype=torch.float32)
        mismatch_logit = float(self.mismatch_head(dist_t).item())
        if self.using_fallback_weights:
            mismatch_logit = (distance - 0.35) * 4.0

        return {
            "active": True,
            "lip_sync_distance": float(np.clip(distance, 0.0, 2.0)),
            "mismatch_logit": float(np.clip(mismatch_logit, -4.0, 4.0)),
            "cosine_similarity": float(cosine),
            "using_fallback_weights": self.using_fallback_weights,
            "motion_energy": motion,
            "audio_envelope": audio_env,
        }

    def forward(self, lips: torch.Tensor, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        viseme = F.normalize(self.visual(lips).mean(dim=1), dim=-1)
        phoneme = F.normalize(self.audio(mel).mean(dim=1), dim=-1)
        cosine = (viseme * phoneme).sum(dim=-1).clamp(-1.0, 1.0)
        distance = 1.0 - cosine
        logit = self.mismatch_head(distance.unsqueeze(-1)).squeeze(-1)
        return {"lip_sync_distance": distance, "mismatch_logit": logit, "cosine_similarity": cosine}

    def _prep_lips(self, lip_rois: Sequence[np.ndarray]) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for roi in lip_rois:
            arr = np.asarray(roi)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            t = torch.from_numpy(arr)[None, ...]
            t = F.interpolate(t.unsqueeze(0), size=(96, 96), mode="bilinear", align_corners=False)
            tensors.append(t.squeeze(0))
        stacked = torch.stack(tensors, dim=0)  # T, 1, 96, 96
        return stacked.unsqueeze(0)

    def _prep_mel(
        self,
        log_mel: Optional[np.ndarray],
        waveform: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        if log_mel is None:
            try:
                import librosa

                spec = librosa.feature.melspectrogram(
                    y=np.asarray(waveform, dtype=np.float32),
                    sr=sample_rate,
                    n_mels=config.N_MELS,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                )
                log_mel = librosa.power_to_db(spec, ref=np.max)
            except Exception:
                n = min(np.asarray(waveform).size, 8192)
                spec = np.abs(np.fft.rfft(np.asarray(waveform[:n], dtype=np.float32)))
                cols = max(1, spec.size // config.N_MELS)
                spec = spec[: config.N_MELS * cols]
                log_mel = spec.reshape(config.N_MELS, cols)
        mel = torch.from_numpy(np.asarray(log_mel, dtype=np.float32))
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        return mel.unsqueeze(0)  # B=1, 1 later added in encoder


def _lip_motion_energy(lip_rois: Sequence[np.ndarray]) -> float:
    if len(lip_rois) < 2:
        return 0.0
    diffs = []
    prev = None
    for roi in lip_rois:
        g = np.asarray(roi, dtype=np.float32)
        if g.ndim == 3:
            g = g.mean(axis=2)
        g = g / (g.max() + 1e-6)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    return float(np.clip(np.mean(diffs) * 8.0, 0.0, 1.5)) if diffs else 0.0


def _audio_envelope_energy(waveform: np.ndarray, sample_rate: int) -> float:
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size < 16:
        return 0.0
    hop = max(1, sample_rate // 30)
    env = np.array([np.sqrt(np.mean(wav[i : i + hop] ** 2)) for i in range(0, len(wav) - hop, hop)])
    if env.size < 2:
        return 0.0
    return float(np.clip(np.std(env) * 6.0, 0.0, 1.5))
