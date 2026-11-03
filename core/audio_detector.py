"""WavLM / Wav2Vec 2.0 + AASIST neural vocoder / speech-spoof detector.

Self-supervised speech embeddings (WavLM Base or Wav2Vec 2.0, loaded from
``/models`` when present) feed an Integrated Spectro-Temporal Graph
Attention classifier inspired by AASIST. Missing checkpoints fall back to
a convolutional waveform encoder plus spectrogram-statistic heuristics
that still emit a logit and fake probability so the app can boot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger("neural_storm.audio")


class ConvWaveformEncoder(nn.Module):
    """Wav2Vec-style 1-D convolutional feature extractor (SSL fallback)."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Parameters: wav ``(B, T)`` → features ``(B, N, C)``."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = wav.unsqueeze(1)
        feat = self.conv(x)
        return feat.transpose(1, 2)


class GraphAttention(nn.Module):
    """Single-head graph attention over spectro-temporal nodes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if adj is not None:
            att = att + adj
        att = torch.softmax(att, dim=-1)
        return self.proj(torch.matmul(att, v))


class AASISTClassifier(nn.Module):
    """Integrated spectro-temporal graph attention (AASIST-style head).

    Temporal nodes capture vocoder periodicity; spectral nodes capture
    neural vocoder high-band corrugation typical of HiFi-GAN / WaveGlow /
    Tacotron pipelines.
    """

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.temporal_gat = GraphAttention(hidden)
        self.spectral_gat = GraphAttention(hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.logit = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters: features ``(B, N, C)`` → (logit ``(B,)``, pooled ``(B, H)``)."""
        h = self.stem(features)
        b, n, c = h.shape
        # Temporal graph: attend along the sequence.
        t_out = self.temporal_gat(h)
        # Spectral graph: pool sequence into a compact frequency-like set of nodes.
        n_spec = min(32, n)
        spec_nodes = F.adaptive_avg_pool1d(h.transpose(1, 2), n_spec).transpose(1, 2)
        s_out = self.spectral_gat(spec_nodes)
        pooled = torch.cat([t_out.mean(dim=1), s_out.mean(dim=1)], dim=-1)
        fused = self.fuse(pooled)
        logit = self.logit(fused).squeeze(-1)
        return logit, fused


class AudioSpoofDetector(nn.Module):
    """WavLM/Wav2Vec encoder + AASIST classifier for synthetic speech."""

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or config.resolve_device())
        self.checkpoint_path = Path(checkpoint or config.CHECKPOINT_AUDIO)
        self.sample_rate = sample_rate
        self.using_fallback_weights = True
        self.ssl_name = "conv_waveform_encoder"

        self.encoder, enc_dim = self._build_ssl_encoder()
        self.classifier = AASISTClassifier(in_dim=enc_dim, hidden=128)
        self._load_or_heuristic_init()
        self.to(self.device)
        self.eval()

    def _build_ssl_encoder(self) -> Tuple[nn.Module, int]:
        ssl = self._try_load_torchaudio_ssl()
        if ssl is not None:
            return ssl
        logger.warning("WavLM / Wav2Vec 2.0 weights not found; using ConvWaveformEncoder.")
        enc = ConvWaveformEncoder(embed_dim=256)
        return enc, enc.embed_dim

    def _try_load_torchaudio_ssl(self) -> Optional[nn.Module]:
        """Load a local WavLM/Wav2Vec bundle without triggering hub downloads."""
        wavlm_ckpt = config.CHECKPOINT_WAVLM
        w2v_ckpt = config.MODELS_DIR / "wav2vec2_base.pt"
        try:
            import torchaudio
        except Exception as exc:
            logger.debug("torchaudio missing: %s", exc)
            return None

        if wavlm_ckpt.is_file() or w2v_ckpt.is_file():
            path = wavlm_ckpt if wavlm_ckpt.is_file() else w2v_ckpt
            try:
                bundle = (
                    torchaudio.pipelines.WAVLM_BASE
                    if path == wavlm_ckpt
                    else torchaudio.pipelines.WAV2VEC2_BASE
                )
                model = bundle.get_model()
                blob = torch.load(path, map_location="cpu")
                state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
                model.load_state_dict(state, strict=False)
                self.ssl_name = "wavlm_base" if path == wavlm_ckpt else "wav2vec2_base"
                logger.info("Loaded SSL speech encoder from %s", path.name)

                class _SSLWrapper(nn.Module):
                    def __init__(self, inner: nn.Module, dim: int) -> None:
                        super().__init__()
                        self.inner = inner
                        self.embed_dim = dim

                    def forward(self, wav: torch.Tensor) -> torch.Tensor:
                        if wav.dim() == 1:
                            wav = wav.unsqueeze(0)
                        feats, _ = self.inner.extract_features(wav)
                        return feats[-1] if isinstance(feats, (list, tuple)) else feats

                dim = int(getattr(model, "encoder_embed_dim", 768))
                return _SSLWrapper(model, dim)
            except Exception as exc:
                logger.warning("SSL checkpoint at %s failed to load: %s", path, exc)
        return None

    def _load_or_heuristic_init(self) -> None:
        if self.checkpoint_path.is_file():
            try:
                blob = torch.load(self.checkpoint_path, map_location="cpu")
                state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
                missing, unexpected = self.load_state_dict(state, strict=False)
                self.using_fallback_weights = False
                logger.info(
                    "Loaded audio checkpoint %s (missing=%d unexpected=%d).",
                    self.checkpoint_path.name,
                    len(missing),
                    len(unexpected),
                )
                return
            except Exception as exc:
                logger.warning("Audio checkpoint load failed (%s); heuristic init.", exc)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        logger.warning(
            "No audio weights at %s — running heuristic/random fallback.",
            self.checkpoint_path,
        )

    def _prepare_wav(self, waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate and wav.size:
            try:
                import torchaudio

                t = torch.from_numpy(wav).unsqueeze(0)
                wav = torchaudio.functional.resample(t, sample_rate, self.sample_rate).squeeze(0).numpy()
            except Exception:
                import librosa

                wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=self.sample_rate)
        max_len = int(config.MAX_AUDIO_SECONDS * self.sample_rate)
        if wav.size > max_len:
            wav = wav[:max_len]
        if wav.size < self.sample_rate:
            wav = np.pad(wav, (0, self.sample_rate - wav.size))
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1e-6:
            wav = wav / peak
        return torch.from_numpy(wav.astype(np.float32))

    @torch.no_grad()
    def infer(
        self,
        waveform: Optional[np.ndarray],
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        log_mel: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return raw audio logit and P(fake)."""
        if waveform is None or np.asarray(waveform).size == 0:
            return {
                "audio_logit": 0.0,
                "fake_probability": 0.5,
                "available": False,
                "using_fallback_weights": self.using_fallback_weights,
                "ssl_name": self.ssl_name,
            }
        wav = self._prepare_wav(waveform, sample_rate).to(self.device)
        out = self.forward(wav.unsqueeze(0))
        model_logit = float(out["audio_logit"])
        heur = heuristic_audio_logit(np.asarray(waveform), sample_rate, log_mel)
        if self.using_fallback_weights:
            logit = 0.40 * model_logit + 0.60 * heur
        else:
            logit = model_logit
        prob = float(torch.sigmoid(torch.tensor(logit / config.TEMPERATURE)))
        return {
            "audio_logit": logit,
            "fake_probability": prob,
            "available": True,
            "using_fallback_weights": self.using_fallback_weights,
            "ssl_name": self.ssl_name,
            "pooled_features": out["pooled"].detach().cpu(),
        }

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        features = self.encoder(wav)
        logit, pooled = self.classifier(features)
        return {
            "audio_logit": logit.reshape(-1)[0] if logit.numel() == 1 else logit,
            "fake_probability": torch.sigmoid(logit / config.TEMPERATURE),
            "pooled": pooled,
            "features": features,
        }


def heuristic_audio_logit(
    waveform: np.ndarray,
    sample_rate: int,
    log_mel: Optional[np.ndarray] = None,
) -> float:
    """Prior for neural vocoders: overly flat high-band residual and crest factor."""
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size < 16:
        return 0.0
    if log_mel is None:
        try:
            import librosa

            spec = librosa.feature.melspectrogram(
                y=wav, sr=sample_rate, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
            )
            log_mel = librosa.power_to_db(spec, ref=np.max)
        except Exception:
            spec = np.abs(np.fft.rfft(wav[: min(len(wav), 16384)]))
            cols = max(1, spec.size // config.N_MELS)
            log_mel = spec[: config.N_MELS * cols].reshape(config.N_MELS, cols)
    lm = np.asarray(log_mel, dtype=np.float32)
    high = lm[int(lm.shape[0] * 0.65) :, :]
    low = lm[: max(1, int(lm.shape[0] * 0.25)), :]
    high_std = float(np.std(high)) + 1e-6
    low_std = float(np.std(low)) + 1e-6
    # Vocoders often suppress high-band variance relative to voiced energy.
    band_ratio = low_std / high_std
    peak = float(np.max(np.abs(wav)) + 1e-9)
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
    crest = peak / rms
    z = (band_ratio - 1.35) * 1.6 + (3.2 - crest) * 0.35
    return float(np.clip(z, -4.0, 4.0))
