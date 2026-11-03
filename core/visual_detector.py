"""Dual-domain visual deepfake detector (EfficientNet-B4 + DCT/SRM + Bi-LSTM).

The spatial stream extracts RGB facial artifacts with EfficientNet-B4
(``timm``). The frequency stream applies a high-pass Discrete Cosine
Transform residual and SRM steganalysis filters to expose generative
grid artifacts. A bidirectional LSTM aggregates frame sequences. When a
PyTorch checkpoint is missing the module still runs using heuristic
high-frequency energy scores plus randomly initialized (eval-mode) heads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger("neural_storm.visual")

ArrayLike = Union[np.ndarray, Sequence[np.ndarray], torch.Tensor]


class SRMConv2d(nn.Module):
    """Frozen Spatial Rich Model high-pass residual filters (3 kernels)."""

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [-1, 2, -2, 2, -1],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[-1, 2, -2, 2, -1],
                 [2, -6, 8, -6, 2],
                 [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2],
                 [-1, 2, -2, 2, -1]],
                [[0, 0, 0, 0, 0],
                 [0, -1, 2, -1, 0],
                 [0, 2, -4, 2, 0],
                 [0, -1, 2, -1, 0],
                 [0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        kernels = kernels / torch.tensor([2.0, 12.0, 4.0]).view(3, 1, 1)
        weight = kernels.unsqueeze(1).repeat(1, 3, 1, 1) / 3.0
        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FallbackSpatialCNN(nn.Module):
    """Lightweight CNN used when ``timm`` / EfficientNet-B4 cannot load."""

    def __init__(self, out_dim: int = 1792) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)
        self.num_features = out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.proj(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class FrequencyStream(nn.Module):
    """High-pass DCT residual + SRM CNN producing a 256-D embedding."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.srm = SRMConv2d()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = high_pass_dct(x)
        srm = self.srm(x)
        fused = torch.cat([residual, srm], dim=1)
        feat = self.net(fused).flatten(1)
        return self.proj(feat)


def high_pass_dct(x: torch.Tensor, keep_frac: float = 0.18) -> torch.Tensor:
    """High-pass residual approximating a DCT grid filter (batched RGB).

    Low-frequency FFT bins (DCT-equivalent energy) are zeroed and inverted.
    Falls back to a 5×5 box residual if the FFT path fails.
    """
    try:
        spec = torch.fft.rfft2(x)
        _b, _c, h, w = x.shape
        rh = max(1, int(h * keep_frac))
        rw = max(1, int((w // 2 + 1) * keep_frac))
        spec = spec.clone()
        spec[..., :rh, :rw] = 0
        rec = torch.fft.irfft2(spec, s=(h, w))
        return rec.real if torch.is_complex(rec) else rec
    except Exception:
        return x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)


class DualDomainVisualDetector(nn.Module):
    """EfficientNet-B4 spatial stream + DCT/SRM frequency stream + Bi-LSTM."""

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or config.resolve_device())
        self.checkpoint_path = Path(checkpoint or config.CHECKPOINT_VISUAL)
        self.using_fallback_weights = True
        self.spatial_dim = 1792
        self.freq_dim = 256

        self.backbone, self.cam_target_name = self._build_backbone(pretrained_backbone)
        self.frequency = FrequencyStream(embed_dim=self.freq_dim)
        fusion_dim = self.spatial_dim + self.freq_dim
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.spatial_head = nn.Linear(self.spatial_dim, 1)
        self.frequency_head = nn.Linear(self.freq_dim, 1)
        self.combined_head = nn.Linear(512, 1)

        self._load_or_heuristic_init()
        self.to(self.device)
        self.eval()

    def _build_backbone(self, pretrained: bool) -> Tuple[nn.Module, str]:
        try:
            import timm

            model = timm.create_model(
                "tf_efficientnet_b4",
                pretrained=bool(pretrained),
                num_classes=0,
                global_pool="avg",
            )
            num_features = int(getattr(model, "num_features", self.spatial_dim))
            self.spatial_dim = num_features
            logger.info("Loaded EfficientNet-B4 backbone (num_features=%s).", num_features)
            return model, "conv_head"
        except Exception as exc:
            logger.warning("timm EfficientNet-B4 unavailable (%s); using FallbackSpatialCNN.", exc)
            fb = FallbackSpatialCNN(out_dim=self.spatial_dim)
            return fb, "features"

    def _load_or_heuristic_init(self) -> None:
        if self.checkpoint_path.is_file():
            try:
                blob = torch.load(self.checkpoint_path, map_location="cpu")
                state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
                missing, unexpected = self.load_state_dict(state, strict=False)
                self.using_fallback_weights = False
                logger.info(
                    "Loaded visual checkpoint %s (missing=%d, unexpected=%d).",
                    self.checkpoint_path.name,
                    len(missing),
                    len(unexpected),
                )
                return
            except Exception as exc:
                logger.warning("Visual checkpoint load failed (%s); heuristic init.", exc)
        self._heuristic_init()
        logger.warning(
            "No visual weights at %s — running heuristic/random fallback.",
            self.checkpoint_path,
        )

    def _heuristic_init(self) -> None:
        """Small-gain random weights so logits stay in a stable range."""
        for module in (self.spatial_head, self.frequency_head, self.combined_head):
            nn.init.xavier_uniform_(module.weight, gain=0.15)
            nn.init.constant_(module.bias, 0.0)

    def gradcam_target(self) -> nn.Module:
        """Return the last convolutional module used by Grad-CAM++."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head  # type: ignore[return-value]
        if hasattr(self.backbone, "features"):
            # Fallback CNN: last Conv2d inside Sequential
            convs = [m for m in self.backbone.features.modules() if isinstance(m, nn.Conv2d)]
            if convs:
                return convs[-1]
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError("No convolutional layer available for Grad-CAM++.")
        return convs[-1]

    def preprocess(self, faces: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert uint8 RGB faces (T, H, W, C) into ImageNet-normalized tensors."""
        if len(faces) == 0:
            raise ValueError("No face crops provided to DualDomainVisualDetector.")
        batch = []
        mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
        for face in faces:
            arr = np.asarray(face)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            if tensor.shape[-2:] != config.FACE_SIZE[::-1] and tensor.shape[-2:] != config.FACE_SIZE:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=config.FACE_SIZE,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensor = (tensor - mean) / std
            batch.append(tensor)
        return torch.stack(batch, dim=0)

    @torch.no_grad()
    def infer(self, faces: Sequence[np.ndarray]) -> Dict[str, Any]:
        """Run the dual-domain detector and merge with heuristic priors."""
        x = self.preprocess(faces).to(self.device)
        out = self.forward(x.unsqueeze(0))
        heur_s, heur_f = heuristic_visual_logits(faces)
        if self.using_fallback_weights:
            # Blend learned-random logits with DSP priors so the demo is informative.
            spatial_logit = 0.35 * float(out["spatial_logit"]) + 0.65 * heur_s
            frequency_logit = 0.35 * float(out["frequency_logit"]) + 0.65 * heur_f
            combined_logit = 0.5 * (spatial_logit + frequency_logit)
            combined_prob = float(torch.sigmoid(torch.tensor(combined_logit / config.TEMPERATURE)))
        else:
            spatial_logit = float(out["spatial_logit"])
            frequency_logit = float(out["frequency_logit"])
            combined_logit = float(out["combined_logit"])
            combined_prob = float(out["combined_prob"])
        return {
            "spatial_logit": spatial_logit,
            "frequency_logit": frequency_logit,
            "combined_logit": combined_logit,
            "combined_prob": combined_prob,
            "sequence_features": out["sequence_features"].detach().cpu(),
            "using_fallback_weights": self.using_fallback_weights,
            "n_frames": len(faces),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            ``(B, T, C, H, W)`` video batch or ``(B, C, H, W)`` still image.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)

        if hasattr(self.backbone, "forward_features"):
            spatial_feat = self.backbone.forward_features(flat)
            if spatial_feat.dim() == 4:
                spatial_feat = spatial_feat.mean(dim=(2, 3))
        else:
            spatial_feat = self.backbone(flat)
            if spatial_feat.dim() == 4:
                spatial_feat = F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1)

        freq_feat = self.frequency(flat)
        spatial_logit = self.spatial_head(spatial_feat)
        frequency_logit = self.frequency_head(freq_feat)

        fused = torch.cat([spatial_feat, freq_feat], dim=-1).view(b, t, -1)
        lstm_out, _ = self.lstm(fused)
        pooled = lstm_out.mean(dim=1)
        combined_logit = self.combined_head(pooled)

        spatial_clip = spatial_logit.view(b, t).mean(dim=1)
        freq_clip = frequency_logit.view(b, t).mean(dim=1)
        combined = combined_logit.reshape(b)

        def _pack(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(-1)[0] if b == 1 else tensor

        return {
            "spatial_logit": _pack(spatial_clip),
            "frequency_logit": _pack(freq_clip),
            "combined_logit": _pack(combined),
            "combined_prob": torch.sigmoid(combined / config.TEMPERATURE),
            "sequence_features": pooled,
            "spatial_features": spatial_feat.view(b, t, -1),
            "freq_features": freq_feat.view(b, t, -1),
        }


def heuristic_visual_logits(faces: Sequence[np.ndarray]) -> Tuple[float, float]:
    """DSP prior: DCT high-band energy and local residual variance."""
    if not faces:
        return 0.0, 0.0
    freq_scores = []
    spatial_scores = []
    for face in faces:
        img = np.asarray(face, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        gray = img.mean(axis=2) if img.ndim == 3 else img
        try:
            from scipy.fftpack import dct

            coeff = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
            h, w = coeff.shape
            high = np.abs(coeff[h // 4 :, w // 4 :]).mean()
            low = np.abs(coeff[: h // 8, : w // 8]).mean() + 1e-6
            ratio = float(high / low)
        except Exception:
            gy, gx = np.gradient(gray)
            ratio = float(np.mean(gx ** 2 + gy ** 2) * 8.0)
        freq_scores.append(ratio)
        # Spatial prior: unnatural smoothness vs. sensor noise.
        lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        var = float(np.var(lap))
        spatial_scores.append(var)

    freq_ratio = float(np.median(freq_scores))
    lap_var = float(np.median(spatial_scores))
    # Generative images often show elevated mid/high DCT structure and low sensor noise.
    z_freq = (freq_ratio - 0.55) * 3.2
    z_spatial = (0.0045 - lap_var) * 180.0
    return float(np.clip(z_spatial, -4.0, 4.0)), float(np.clip(z_freq, -4.0, 4.0))
