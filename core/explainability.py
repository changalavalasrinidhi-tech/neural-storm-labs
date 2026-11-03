"""Grad-CAM++ spatial heatmaps, spectrogram anomaly overlays, and narratives.

Produces forensic visualizations grounded in the same logits used by
:class:`core.fusion.CalibratedEvidenceFusion`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

logger = logging.getLogger("neural_storm.explain")


@dataclass
class ExplanationBundle:
    """Explainability artifacts for the Streamlit diagnostic pane."""

    heatmap_rgb: Optional[np.ndarray]
    overlay_rgb: Optional[np.ndarray]
    original_face: Optional[np.ndarray]
    spectrogram_overlay: Optional[np.ndarray]
    spectrogram_raw: Optional[np.ndarray]
    narrative: str
    cam_available: bool
    metadata: Dict[str, Any]


class ForensicExplainer:
    """Grad-CAM++ on EfficientNet-B4 plus spectrogram anomaly highlighting."""

    def __init__(self, visual_detector: Optional[nn.Module] = None) -> None:
        self.visual_detector = visual_detector

    def explain(
        self,
        faces: Sequence[np.ndarray],
        log_mel: Optional[np.ndarray],
        fusion: Any,
        visual_out: Optional[Dict[str, Any]] = None,
        audio_out: Optional[Dict[str, Any]] = None,
        sync_out: Optional[Dict[str, Any]] = None,
        target_index: Optional[int] = None,
    ) -> ExplanationBundle:
        original = None
        heatmap = None
        overlay = None
        cam_ok = False
        if faces:
            idx = int(target_index) if target_index is not None else max(len(faces) // 2, 0)
            idx = min(idx, len(faces) - 1)
            original = np.asarray(faces[idx]).copy()
            try:
                heatmap, overlay = self.gradcam_plus_plus(faces[idx])
                cam_ok = heatmap is not None
            except Exception as exc:
                logger.warning("Grad-CAM++ failed (%s); using frequency residual overlay.", exc)
                heatmap, overlay = self._frequency_fallback_heatmap(faces[idx])

        spec_raw, spec_overlay = self.spectrogram_anomaly_overlay(log_mel) if log_mel is not None else (None, None)
        narrative = self.build_narrative(fusion, visual_out, audio_out, sync_out)
        return ExplanationBundle(
            heatmap_rgb=heatmap,
            overlay_rgb=overlay,
            original_face=original,
            spectrogram_overlay=spec_overlay,
            spectrogram_raw=spec_raw,
            narrative=narrative,
            cam_available=cam_ok,
            metadata={"n_faces": len(faces), "has_mel": log_mel is not None},
        )

    def gradcam_plus_plus(self, face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Grad-CAM++ over the last conv layer of the spatial backbone."""
        detector = self.visual_detector
        if detector is None:
            return self._frequency_fallback_heatmap(face)

        detector.eval()
        target = detector.gradcam_target()
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            activations.append(out)

        def bwd_hook(_module: nn.Module, _gin: Any, gout: Any) -> None:
            gradients.append(gout[0])

        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)
        try:
            x = detector.preprocess([face]).to(detector.device)
            x = x.unsqueeze(0)
            x.requires_grad_(True)
            detector.zero_grad(set_to_none=True)
            out = detector.forward(x)
            logit = out["combined_logit"]
            if not torch.is_tensor(logit):
                logit = torch.tensor(float(logit), device=x.device, requires_grad=True)
            if logit.numel() > 1:
                logit = logit.reshape(-1)[0]
            logit.backward(retain_graph=False)
            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks produced no tensors.")
            act = activations[0].detach()
            grad = gradients[0].detach()
            cam = self._compute_gradcam_pp(act, grad)
        finally:
            h1.remove()
            h2.remove()
            detector.zero_grad(set_to_none=True)

        cam_np = cam.squeeze().cpu().numpy()
        heatmap = _colorize_heatmap(cam_np, face.shape[:2])
        overlay = _blend(face, heatmap, alpha=0.45)
        return heatmap, overlay

    @staticmethod
    def _compute_gradcam_pp(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """Chattopadhyay et al. Grad-CAM++ weights."""
        # activations / gradients: (B, K, H, W)
        if activations.dim() == 2:
            activations = activations[:, :, None, None]
            gradients = gradients[:, :, None, None]
        grads = gradients
        grads2 = grads ** 2
        grads3 = grads ** 3
        sum_act = activations.flatten(2).sum(-1).view(*activations.shape[:2], 1, 1) + 1e-8
        denom = 2.0 * grads2 + sum_act * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads2 / denom
        weights = (alpha * torch.relu(grads)).flatten(2).sum(-1)  # (B, K)
        weights = weights.view(weights.shape[0], weights.shape[1], 1, 1)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        b, _, h, w = cam.shape
        cam = cam.view(b, -1)
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
        return cam.view(b, 1, h, w)

    def spectrogram_anomaly_overlay(
        self, log_mel: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Highlight synthetic high-frequency bands on the log-mel spectrogram."""
        if log_mel is None:
            return None, None
        lm = np.asarray(log_mel, dtype=np.float32)
        if lm.ndim == 1:
            lm = lm[:, None]
        raw = _spec_to_rgb(lm)
        overlay = raw.copy()
        # Anomaly: local energy far above the per-band median (vocoder corrugation).
        band_med = np.median(lm, axis=1, keepdims=True)
        band_mad = np.median(np.abs(lm - band_med), axis=1, keepdims=True) + 1e-6
        z = (lm - band_med) / (1.4826 * band_mad)
        high_band = np.zeros_like(lm, dtype=bool)
        high_band[int(lm.shape[0] * 0.55) :, :] = True
        mask = (z > 2.2) & high_band
        boxes = _mask_to_boxes(mask)
        overlay = _draw_boxes(overlay, boxes, color=(255, 64, 96))
        heat = np.clip(z, 0, 6) / 6.0
        heat_rgb = _colorize_heatmap(heat, overlay.shape[:2])
        overlay = _blend(overlay, heat_rgb, alpha=0.35)
        return raw, overlay

    def build_narrative(
        self,
        fusion: Any,
        visual_out: Optional[Dict[str, Any]],
        audio_out: Optional[Dict[str, Any]],
        sync_out: Optional[Dict[str, Any]],
    ) -> str:
        """Strict logit-grounded text: every clause cites a numeric evidence term."""
        visual_out = visual_out or {}
        audio_out = audio_out or {}
        sync_out = sync_out or {}
        w = fusion.weights
        z = fusion.stream_logits
        lines: List[str] = []
        lines.append("NEURAL STORM FT-02  ·  LOGIT-GROUNDED FORENSIC NARRATIVE")
        lines.append("=" * 72)
        lines.append(
            f"Fused logit  z̃ = w_v·z_v + w_a·z_a + w_s·z_s = "
            f"{w['w_v']:.3f}·{z['visual']:+.4f} + {w['w_a']:.3f}·{z['audio']:+.4f} + "
            f"{w['w_s']:.3f}·{z['sync']:+.4f} = {fusion.fused_logit:+.4f}"
        )
        t = fusion.metadata.get("temperature", config.TEMPERATURE)
        a = fusion.metadata.get("platt_scale", 1.0)
        b = fusion.metadata.get("platt_bias", 0.0)
        lines.append(
            f"Temperature / Platt calibration:  p = σ((a·z̃ + b) / T) with "
            f"a={a:.3f}, b={b:.3f}, T={t:.3f}  →  P(synthetic) = {fusion.probability:.4f} "
            f"({fusion.probability_pct:.1f}%)."
        )
        lo, hi = fusion.confidence_interval
        lines.append(
            f"95% interval on calibrated probability: [{lo:.3f}, {hi:.3f}].  "
            f"Epistemic disagreement = {fusion.epistemic_uncertainty:.3f} "
            f"(threshold {config.EPISTEMIC_UNCERTAINTY_THRESHOLD:.2f})."
        )
        band = fusion.metadata.get("uncertain_band", (config.UNCERTAIN_SCORE_LOW, config.UNCERTAIN_SCORE_HIGH))
        lines.append(
            f"Uncertainty policy: flag if {band[0]:.2f} ≤ p ≤ {band[1]:.2f} "
            f"or epistemic > {config.EPISTEMIC_UNCERTAINTY_THRESHOLD:.2f}.  "
            f"Verdict: {fusion.label}."
        )
        lines.append("")
        lines.append("Visual stream (EfficientNet-B4 spatial + DCT/SRM frequency)")
        lines.append("-" * 72)
        z_s = visual_out.get("spatial_logit")
        z_f = visual_out.get("frequency_logit")
        if z_s is not None and z_f is not None:
            lines.append(
                f"  Spatial logit z_spatial = {float(z_s):+.4f}; "
                f"frequency logit z_freq = {float(z_f):+.4f}; "
                f"combined visual z_v = {z['visual']:+.4f} "
                f"(P_v = {fusion.stream_probs.get('visual', float('nan')):.3f})."
            )
            contrib_v = w["w_v"] * z["visual"]
            lines.append(
                f"  Quality-adjusted weight w_v = {w['w_v']:.3f} "
                f"(Laplacian variance SQI = {fusion.sqi.get('visual_laplacian_var', 0.0):.5f}); "
                f"visual contribution to z̃ = {contrib_v:+.4f}."
            )
            if float(z_f) > 0:
                lines.append(
                    "  Positive frequency logit indicates elevated high-pass DCT / SRM residual "
                    "energy consistent with diffusion or GAN grid artifacts."
                )
            else:
                lines.append(
                    "  Non-positive frequency logit indicates residual energy closer to natural "
                    "sensor noise than to periodic generative spectra."
                )
            if visual_out.get("using_fallback_weights"):
                lines.append("  NOTE: visual checkpoint missing — logits include DSP heuristic priors.")
        else:
            lines.append("  Visual stream inactive (no face ROI). w_v redistributed.")

        lines.append("")
        lines.append("Audio stream (WavLM / Wav2Vec 2.0 + AASIST)")
        lines.append("-" * 72)
        if audio_out.get("available"):
            lines.append(
                f"  Encoder = {audio_out.get('ssl_name', 'unknown')}; "
                f"audio logit z_a = {z['audio']:+.4f} "
                f"(P_a = {fusion.stream_probs.get('audio', float('nan')):.3f})."
            )
            contrib_a = w["w_a"] * z["audio"]
            lines.append(
                f"  Quality-adjusted weight w_a = {w['w_a']:.3f} "
                f"(SNR SQI = {fusion.sqi.get('audio_snr_db', 0.0):.2f} dB); "
                f"audio contribution to z̃ = {contrib_a:+.4f}."
            )
            if z["audio"] > 0:
                lines.append(
                    "  Positive audio logit is consistent with neural vocoder synthesis artifacts "
                    "(HiFi-GAN / WaveGlow / Tacotron-class spectro-temporal graphs)."
                )
            else:
                lines.append(
                    "  Non-positive audio logit is consistent with organic glottal residual structure."
                )
            if audio_out.get("using_fallback_weights"):
                lines.append("  NOTE: audio checkpoint missing — logits include spectrogram-statistic priors.")
        else:
            lines.append("  Audio stream inactive (no waveform). w_a = 0.")

        lines.append("")
        lines.append("Lip-sync stream (SyncNet phoneme–viseme cosine distance)")
        lines.append("-" * 72)
        if sync_out.get("active"):
            lines.append(
                f"  Cosine similarity = {float(sync_out.get('cosine_similarity', 0.0)):.4f}; "
                f"lip-sync distance d = {float(sync_out.get('lip_sync_distance', 0.0)):.4f}; "
                f"mismatch logit z_s = {z['sync']:+.4f} "
                f"(P_s = {fusion.stream_probs.get('sync', float('nan')):.3f})."
            )
            contrib_s = w["w_s"] * z["sync"]
            lines.append(
                f"  Weight w_s = {w['w_s']:.3f}; sync contribution to z̃ = {contrib_s:+.4f}."
            )
            if z["sync"] > 0:
                lines.append(
                    "  Positive mismatch logit indicates phoneme embeddings diverging from viseme "
                    "motion vectors (classic face-swap / reenactment desynchronization)."
                )
            else:
                lines.append("  Phoneme–viseme alignment is within the natural-speech cosine band.")
        else:
            lines.append("  Sync stream inactive (requires both facial tracks and a speech waveform).")

        lines.append("")
        lines.append(
            f"Final classification: {fusion.label}  |  "
            f"P(synthetic) = {fusion.probability_pct:.1f}%  |  "
            f"CI[{lo * 100:.1f}%, {hi * 100:.1f}%]"
        )
        return "\n".join(lines)

    @staticmethod
    def _frequency_fallback_heatmap(face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = np.asarray(face, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        gray = img.mean(axis=2) if img.ndim == 3 else img
        try:
            from scipy.fftpack import dct, idct

            coeff = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
            h, w = coeff.shape
            coeff[: h // 6, : w // 6] = 0
            rec = idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")
            cam = np.abs(rec)
        except Exception:
            gy, gx = np.gradient(gray)
            cam = gx ** 2 + gy ** 2
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        heatmap = _colorize_heatmap(cam, face.shape[:2])
        return heatmap, _blend(face, heatmap, alpha=0.45)


def _colorize_heatmap(cam: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    h, w = size_hw[:2]
    try:
        import cv2

        cam_u8 = cv2.resize((cam * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h)))
        t = resized.astype(np.float32) / 255.0
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _blend(base: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = np.asarray(base)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    heat = np.asarray(heat)
    if heat.shape[:2] != img.shape[:2]:
        try:
            import cv2

            heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
        except Exception:
            from PIL import Image

            heat = np.array(Image.fromarray(heat).resize((img.shape[1], img.shape[0])))
    blend = (1.0 - alpha) * img.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


def _spec_to_rgb(log_mel: np.ndarray) -> np.ndarray:
    lm = np.asarray(log_mel, dtype=np.float32)
    lm = lm - lm.min()
    lm = lm / (lm.max() + 1e-8)
    u8 = np.flipud((lm * 255).astype(np.uint8))
    try:
        import cv2

        color = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if h < 180:
            rgb = cv2.resize(rgb, (max(w * 2, 480), 220), interpolation=cv2.INTER_NEAREST)
        return rgb
    except Exception:
        from PIL import Image

        t = u8.astype(np.float32) / 255.0
        rgb = np.stack([t, t * 0.4, t * 0.7], axis=-1)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        return np.array(img.resize((max(u8.shape[1] * 2, 480), 220)))


def _mask_to_boxes(mask: np.ndarray, min_area: int = 8) -> List[Tuple[int, int, int, int]]:
    m = np.flipud(mask.astype(bool))
    boxes: List[Tuple[int, int, int, int]] = []
    try:
        import cv2

        contours, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                boxes.append((x, y, x + w, y + h))
        return boxes[:24]
    except Exception:
        ys, xs = np.where(m)
        if ys.size == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]


def _draw_boxes(image: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xa, ya = int(max(0, min(w - 1, x1))), int(max(0, min(h - 1, y1)))
        xb, yb = int(max(0, min(w - 1, x2))), int(max(0, min(h - 1, y2)))
        canvas[ya : ya + 1, xa:xb] = color
        canvas[yb : yb + 1, xa:xb] = color
        canvas[ya:yb, xa : xa + 1] = color
        canvas[ya:yb, xb : xb + 1] = color
    return canvas
