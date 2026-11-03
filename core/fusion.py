"""Quality-weighted logit fusion, temperature calibration, and uncertainty.

Implements:
    Logit_fused = w_v * z_v + w_a * z_a + w_s * z_s

Weights are re-balanced from Signal Quality Index (visual Laplacian
variance, audio SNR). Logits are calibrated with temperature scaling
(T = 1.5 by default; Platt-style affine calibration when a scaler
checkpoint is present). Epistemic disagreement across streams plus the
calibrated score band decide the forensic label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import config

logger = logging.getLogger("neural_storm.fusion")


@dataclass
class FusionResult:
    """Calibrated multi-stream verdict."""

    fused_logit: float
    probability: float
    probability_pct: float
    label: str
    confidence_interval: Tuple[float, float]
    epistemic_uncertainty: float
    weights: Dict[str, float]
    sqi: Dict[str, float]
    stream_logits: Dict[str, float]
    stream_probs: Dict[str, float]
    uncertain: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemperatureScaler(nn.Module):
    """Platt / temperature scaling: ``p = sigmoid((a * z + b) / T)``."""

    def __init__(self, temperature: float = config.TEMPERATURE) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, logit: torch.Tensor) -> torch.Tensor:
        t = self.temperature.clamp(min=0.5, max=5.0)
        return (self.scale * logit + self.bias) / t

    def probability(self, logit: float) -> float:
        z = self.forward(torch.tensor(float(logit)))
        return float(torch.sigmoid(z).item())


class CalibratedEvidenceFusion:
    """Dynamic quality-weighted fusion of visual, audio, and sync logits."""

    def __init__(
        self,
        temperature: float = config.TEMPERATURE,
        base_weights: Optional[Dict[str, float]] = None,
        uncertain_low: float = config.UNCERTAIN_SCORE_LOW,
        uncertain_high: float = config.UNCERTAIN_SCORE_HIGH,
        epistemic_threshold: float = config.EPISTEMIC_UNCERTAINTY_THRESHOLD,
        scaler_checkpoint: Optional[Path] = None,
    ) -> None:
        self.base_weights = dict(base_weights or config.DEFAULT_FUSION_WEIGHTS)
        self.uncertain_low = float(uncertain_low)
        self.uncertain_high = float(uncertain_high)
        self.epistemic_threshold = float(epistemic_threshold)
        self.scaler = TemperatureScaler(temperature=temperature)
        self._load_scaler(scaler_checkpoint or config.CHECKPOINT_TEMPERATURE)

    def _load_scaler(self, path: Path) -> None:
        if not Path(path).is_file():
            return
        try:
            blob = torch.load(path, map_location="cpu")
            state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
            self.scaler.load_state_dict(state, strict=False)
            logger.info("Loaded temperature scaler from %s", path)
        except Exception as exc:
            logger.warning("Temperature scaler load failed: %s", exc)

    def fuse(
        self,
        z_visual: Optional[float],
        z_audio: Optional[float],
        z_sync: Optional[float],
        faces: Optional[Sequence[np.ndarray]] = None,
        waveform: Optional[np.ndarray] = None,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        sync_active: bool = False,
        sensitivity: float = 0.5,
    ) -> FusionResult:
        """Fuse available stream logits into a calibrated forensic verdict.

        Parameters
        ----------
        sensitivity:
            UI slider in ``[0, 1]``. Higher values shrink the uncertain
            band (more decisive labels).
        """
        sqi_v = self.visual_sqi(faces)
        sqi_a = self.audio_sqi(waveform, sample_rate)
        weights = self._dynamic_weights(sqi_v, sqi_a, z_visual, z_audio, z_sync, sync_active)

        z_v = 0.0 if z_visual is None else float(z_visual)
        z_a = 0.0 if z_audio is None else float(z_audio)
        z_s = 0.0 if z_sync is None else float(z_sync)

        fused = weights["w_v"] * z_v + weights["w_a"] * z_a + weights["w_s"] * z_s
        probability = self.scaler.probability(fused)
        probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))

        stream_logits = {"visual": z_v, "audio": z_a, "sync": z_s}
        stream_probs: Dict[str, float] = {}
        available_probs = []
        t = float(self.scaler.temperature.detach())
        if weights["w_v"] > 0:
            stream_probs["visual"] = float(torch.sigmoid(torch.tensor(z_v / t)).item())
            available_probs.append(stream_probs["visual"])
        if weights["w_a"] > 0:
            stream_probs["audio"] = float(torch.sigmoid(torch.tensor(z_a / t)).item())
            available_probs.append(stream_probs["audio"])
        if weights["w_s"] > 0:
            stream_probs["sync"] = float(torch.sigmoid(torch.tensor(z_s / t)).item())
            available_probs.append(stream_probs["sync"])

        epistemic = self._epistemic(available_probs, probability)
        low, high = self._adjust_band(sensitivity)
        uncertain = (low <= probability <= high) or (epistemic > self.epistemic_threshold)
        label = self._label(probability, uncertain)
        ci = self._confidence_interval(probability, epistemic)

        return FusionResult(
            fused_logit=float(fused),
            probability=probability,
            probability_pct=float(probability * 100.0),
            label=label,
            confidence_interval=ci,
            epistemic_uncertainty=float(epistemic),
            weights=weights,
            sqi={"visual_laplacian_var": sqi_v, "audio_snr_db": sqi_a},
            stream_logits=stream_logits,
            stream_probs=stream_probs,
            uncertain=uncertain,
            metadata={
                "temperature": float(self.scaler.temperature.detach()),
                "platt_scale": float(self.scaler.scale.detach()),
                "platt_bias": float(self.scaler.bias.detach()),
                "uncertain_band": (low, high),
                "sensitivity": sensitivity,
            },
        )

    def visual_sqi(self, faces: Optional[Sequence[np.ndarray]]) -> float:
        """Laplacian variance of face crops (higher = sharper / more usable)."""
        if not faces:
            return 0.0
        vars_: List[float] = []
        for face in faces:
            img = np.asarray(face, dtype=np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            gray = img.mean(axis=2) if img.ndim == 3 else img
            lap = (
                gray[1:-1, 1:-1] * 4
                - gray[:-2, 1:-1]
                - gray[2:, 1:-1]
                - gray[1:-1, :-2]
                - gray[1:-1, 2:]
            )
            vars_.append(float(np.var(lap)))
        return float(np.median(vars_)) if vars_ else 0.0

    def audio_sqi(self, waveform: Optional[np.ndarray], sample_rate: int) -> float:
        """Approximate SNR in dB from percentile noise floor vs. RMS."""
        if waveform is None:
            return 0.0
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wav.size < 32:
            return 0.0
        power = wav ** 2
        noise = float(np.percentile(power, 15) + 1e-12)
        signal = float(np.mean(power) + 1e-12)
        snr = 10.0 * np.log10(signal / noise)
        return float(np.clip(snr, -10.0, 40.0))

    def _dynamic_weights(
        self,
        sqi_v: float,
        sqi_a: float,
        z_v: Optional[float],
        z_a: Optional[float],
        z_s: Optional[float],
        sync_active: bool,
    ) -> Dict[str, float]:
        w_v = self.base_weights["w_v"] if z_v is not None else 0.0
        w_a = self.base_weights["w_a"] if z_a is not None else 0.0
        w_s = self.base_weights["w_s"] if (sync_active and z_s is not None) else 0.0

        # Reduce visual weight on heavy blur (typical Laplacian var of a sharp
        # 224px face is ~0.002–0.02 for [0,1] images).
        if w_v > 0:
            blur_factor = float(np.clip(sqi_v / 0.006, 0.25, 1.15))
            w_v *= blur_factor
        if w_a > 0:
            snr_factor = float(np.clip((sqi_a - 2.0) / 18.0, 0.25, 1.15))
            w_a *= snr_factor

        total = w_v + w_a + w_s
        if total <= 1e-8:
            # Degenerate: no evidence — equal dummy mass on visual to keep a score.
            return {"w_v": 1.0, "w_a": 0.0, "w_s": 0.0}
        return {"w_v": w_v / total, "w_a": w_a / total, "w_s": w_s / total}

    @staticmethod
    def _epistemic(stream_probs: Sequence[float], fused_p: float) -> float:
        if len(stream_probs) >= 2:
            disagreement = float(np.std(np.asarray(stream_probs, dtype=np.float64)))
        elif len(stream_probs) == 1:
            disagreement = 0.0
        else:
            disagreement = 0.5
        # Distance from a confident pole, scaled.
        polarity = float(1.0 - 2.0 * abs(fused_p - 0.5))
        return float(np.clip(0.65 * disagreement + 0.35 * polarity * 0.5, 0.0, 1.0))

    def _adjust_band(self, sensitivity: float) -> Tuple[float, float]:
        s = float(np.clip(sensitivity, 0.0, 1.0))
        # High sensitivity shrinks [0.40, 0.60] toward [0.47, 0.53].
        # Low sensitivity widens toward [0.32, 0.68].
        width = (self.uncertain_high - self.uncertain_low) * (1.15 - 0.7 * s)
        center = 0.5
        low = float(np.clip(center - width / 2.0, 0.05, 0.49))
        high = float(np.clip(center + width / 2.0, 0.51, 0.95))
        return low, high

    @staticmethod
    def _label(probability: float, uncertain: bool) -> str:
        if uncertain:
            return config.LABEL_UNCERTAIN
        if probability >= 0.5:
            return config.LABEL_SYNTHETIC
        return config.LABEL_AUTHENTIC

    @staticmethod
    def _confidence_interval(p: float, epistemic: float) -> Tuple[float, float]:
        half = 1.96 * (0.08 + 0.35 * epistemic) * np.sqrt(max(p * (1.0 - p), 1e-4))
        lo = float(np.clip(p - half, 0.0, 1.0))
        hi = float(np.clip(p + half, 0.0, 1.0))
        return lo, hi
