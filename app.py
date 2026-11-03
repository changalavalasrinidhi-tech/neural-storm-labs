"""Neural Storm FT-02 — Streamlit forensic dashboard and pipeline orchestrator.

Launches with heuristic/random fallbacks when ``/models`` checkpoints are
absent so the UI is usable out of the box.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

config.ensure_runtime_dirs()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("neural_storm.app")

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Streamlit is required to launch the dashboard: {exc}") from exc

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is in requirements.txt
    go = None  # type: ignore[assignment]

import ui_labs
from labs import demo as labs_demo
from ui_labs import show_image, show_plot

# The deep pipeline needs PyTorch & friends. If they are missing the Labs tabs must still work.
try:
    from core.ingestion import IngestedMedia, MediaIngestor
    from core.visual_detector import DualDomainVisualDetector
    from core.audio_detector import AudioSpoofDetector
    from core.sync_engine import LipSyncEngine
    from core.fusion import CalibratedEvidenceFusion, FusionResult
    from core.explainability import ExplanationBundle, ForensicExplainer

    CORE_OK, CORE_ERR = True, ""
except Exception as _core_exc:  # noqa: BLE001
    CORE_OK, CORE_ERR = False, f"{type(_core_exc).__name__}: {_core_exc}"
    logger.warning("Deep pipeline unavailable: %s", CORE_ERR)


st.set_page_config(
    page_title="Neural Storm FT-02",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#f4f7fb; --muted:#a8b4c7; --panel:#111a2b; --line:rgba(174,194,226,.16); --mint:#61f2c2; --violet:#a78bfa; --amber:#ffc86b; }
.stApp { background: radial-gradient(900px 520px at 8% -5%, rgba(99,102,241,.22), transparent 62%), radial-gradient(700px 450px at 100% 0%, rgba(45,212,191,.13), transparent 60%), linear-gradient(135deg,#080d18 0%,#0b1220 48%,#101426 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10182a 0%,#0b1220 100%); border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:rgba(8,13,24,.75); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#f7f9ff; }
h1 { font-size:2.7rem !important; }
.stMarkdown p, .stCaption { color:#d1d9e8; }
.ns-hero { padding:1.45rem 1.65rem; border-radius:22px; border:1px solid rgba(167,139,250,.27); background:linear-gradient(115deg,rgba(99,102,241,.18),rgba(17,26,43,.78) 48%,rgba(45,212,191,.10)); box-shadow:0 18px 55px rgba(0,0,0,.20); margin-bottom:1.2rem; }
.ns-eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:var(--mint); }
.ns-subtitle { color:#b9c5d8; font-size:1rem; line-height:1.65; max-width:900px; }
.ns-badge { display:inline-block; padding:.42rem .85rem; border-radius:999px; font-weight:700; font-size:.82rem; letter-spacing:.02em; }
.ns-auth { background:rgba(52,211,153,.13); color:#7df6c8; border:1px solid rgba(52,211,153,.4); }
.ns-fake { background:rgba(251,113,133,.13); color:#ff9eae; border:1px solid rgba(251,113,133,.42); }
.ns-unc { background:rgba(251,191,36,.13); color:#ffda83; border:1px solid rgba(251,191,36,.4); }
.ns-card { background:linear-gradient(145deg,rgba(22,33,54,.92),rgba(13,21,36,.9)); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 12px 30px rgba(0,0,0,.12); }
.ns-muted { color:var(--muted) !important; font-size:.9rem; line-height:1.55; }
[data-testid="stMetric"] { background:rgba(17,26,43,.76); border:1px solid var(--line); border-radius:15px; padding:.8rem 1rem; }
[data-testid="stMetricValue"] { color:#f4f7fb; }
.stButton>button, [data-testid="stDownloadButton"] button { border-radius:12px; border:1px solid rgba(97,242,194,.38); background:linear-gradient(100deg,#42d9b0,#83f0d1); color:#07131b; font-weight:700; transition:all .2s ease; }
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover { border-color:#b6ffe9; box-shadow:0 0 22px rgba(97,242,194,.18); transform:translateY(-1px); color:#07131b; }
[data-testid="stFileUploader"] { border:1px dashed rgba(167,139,250,.48); border-radius:16px; background:rgba(99,102,241,.06); padding:.4rem; }
[data-testid="stTabs"] button { font-weight:600; }
.ns-input { padding:1.1rem 1.3rem; border-radius:20px; border:1px solid rgba(97,242,194,.38); background:linear-gradient(120deg,rgba(45,212,191,.12),rgba(17,26,43,.82) 55%,rgba(99,102,241,.14)); margin:.2rem 0 1rem 0; }
.ns-input-title { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; color:#f7f9ff; margin:.35rem 0 .15rem 0; word-break:break-word; }
.ns-pills { margin:.45rem 0 .2rem 0; }
.ns-pill { display:inline-block; padding:.28rem .8rem; margin:0 .4rem .3rem 0; border-radius:999px; font-size:.8rem; font-weight:600; color:#7f8ca3; border:1px dashed rgba(174,194,226,.28); }
.ns-pill-on { color:#07131b; background:linear-gradient(100deg,#42d9b0,#83f0d1); border:1px solid #b6ffe9; box-shadow:0 0 18px rgba(97,242,194,.25); }
.ns-chip { display:inline-block; padding:.22rem .65rem; margin:.25rem .35rem 0 0; border-radius:10px; font-size:.78rem; font-weight:600; }
.ns-chip-on { color:#7df6c8; background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.38); }
.ns-chip-off { color:#8995aa; background:rgba(148,163,184,.08); border:1px solid rgba(148,163,184,.25); text-decoration:line-through; text-decoration-color:rgba(148,163,184,.45); }
.ns-bar { height:8px; border-radius:99px; background:rgba(148,163,184,.18); overflow:hidden; margin:.35rem 0 .5rem 0; }
.ns-bar>div { height:100%; border-radius:99px; }
hr { border-color:var(--line); }
textarea, .stCode, code { font-family:'IBM Plex Mono',Consolas,monospace !important; }
</style>
"""


def _badge_class(label: str) -> str:
    if label == config.LABEL_AUTHENTIC:
        return "ns-auth"
    if label == config.LABEL_SYNTHETIC:
        return "ns-fake"
    return "ns-unc"


@st.cache_resource(show_spinner=False)
def load_engines(device_pref: str) -> Dict[str, Any]:
    """Instantiate detectors once per device selection (fallback-safe)."""
    device = config.resolve_device(device_pref)
    status: Dict[str, str] = {}
    visual = audio = sync = None
    try:
        visual = DualDomainVisualDetector(device=device)
        status["visual"] = "fallback" if visual.using_fallback_weights else "checkpoint"
    except Exception as exc:
        status["visual"] = f"error: {exc}"
        logger.exception("Visual detector init failed")
    try:
        audio = AudioSpoofDetector(device=device)
        status["audio"] = "fallback" if audio.using_fallback_weights else "checkpoint"
    except Exception as exc:
        status["audio"] = f"error: {exc}"
        logger.exception("Audio detector init failed")
    try:
        sync = LipSyncEngine(device=device)
        status["sync"] = "fallback" if sync.using_fallback_weights else "checkpoint"
    except Exception as exc:
        status["sync"] = f"error: {exc}"
        logger.exception("Sync engine init failed")
    fusion = CalibratedEvidenceFusion()
    explainer = ForensicExplainer(visual_detector=visual)
    return {
        "device": device,
        "visual": visual,
        "audio": audio,
        "sync": sync,
        "fusion": fusion,
        "explainer": explainer,
        "status": status,
    }


def _gauge(probability_pct: float, label: str) -> go.Figure:
    color = "#7dffb1" if label == config.LABEL_AUTHENTIC else "#ff8da0" if label == config.LABEL_SYNTHETIC else "#ffd56a"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(probability_pct),
            number={"suffix": "%", "font": {"size": 38, "color": color}},
            title={"text": "P(synthetic)", "font": {"size": 16, "color": "#9bb0c9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9bb0c9"},
                "bar": {"color": color},
                "bgcolor": "#101824",
                "borderwidth": 1,
                "bordercolor": "#243044",
                "steps": [
                    {"range": [0, 40], "color": "#12351f"},
                    {"range": [40, 60], "color": "#3a2e10"},
                    {"range": [60, 100], "color": "#3a1218"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75, "value": probability_pct},
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dce7f5"},
    )
    return fig


def _save_upload(uploaded: Any) -> Path:
    suffix = Path(uploaded.name).suffix.lower() or ".bin"
    fd, name = tempfile.mkstemp(prefix="ns_upload_", suffix=suffix, dir=str(config.TEMP_DIR))
    os.close(fd)  # the original code leaked this descriptor
    tmp = Path(name)
    tmp.write_bytes(uploaded.getbuffer())
    return tmp


def _sweep_temp(max_age_s: float = 3600.0) -> None:
    """Remove stale upload files / job folders from temp/ (best effort)."""
    now = time.time()
    try:
        for p in config.TEMP_DIR.iterdir():
            if p.name.startswith(("ns_upload_", "job_")) and now - p.stat().st_mtime > max_age_s:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    except OSError:
        pass


def run_pipeline(
    path: Path,
    engines: Dict[str, Any],
    sensitivity: float,
    progress: Any,
    status_box: Any,
) -> Tuple[IngestedMedia, Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], FusionResult, ExplanationBundle]:
    def _tick(frac: float, msg: str) -> None:
        progress.progress(min(max(frac, 0.0), 1.0), text=msg)
        status_box.markdown(f"<div class='ns-muted'>{msg}</div>", unsafe_allow_html=True)

    _tick(0.05, "1/6  Inspecting media format and routing modality…")
    max_frames = config.MAX_FRAMES_CUDA if engines["device"] == "cuda" else config.MAX_FRAMES_CPU
    ingestor = MediaIngestor(max_frames=max_frames)
    try:
        media = ingestor.ingest(path)
    except Exception:
        ingestor.close()
        raise

    visual_out: Dict[str, Any] = {}
    audio_out: Optional[Dict[str, Any]] = None
    sync_out: Dict[str, Any] = {"active": False}

    _tick(0.22, "2/6  Dual-domain visual stream (EfficientNet-B4 + DCT/SRM)…")
    z_v = None
    if engines["visual"] is not None and media.faces:
        try:
            visual_out = engines["visual"].infer(media.faces)
            z_v = float(visual_out["combined_logit"])
        except Exception as exc:
            logger.exception("Visual inference failed")
            visual_out = {"error": str(exc), "using_fallback_weights": True}

    _tick(0.44, "3/6  Audio spoof stream (WavLM / AASIST)…")
    z_a = None
    if engines["audio"] is not None and media.has_audio:
        try:
            audio_out = engines["audio"].infer(media.audio_waveform, media.sample_rate, media.log_mel)
            if audio_out.get("available"):
                z_a = float(audio_out["audio_logit"])
        except Exception as exc:
            logger.exception("Audio inference failed")
            audio_out = {"error": str(exc), "available": False}

    _tick(0.62, "4/6  SyncNet phoneme–viseme alignment…")
    z_s = None
    sync_active = False
    if engines["sync"] is not None and media.has_speech_and_faces:
        try:
            sync_out = engines["sync"].infer(
                lip_rois=media.lip_rois or media.faces,
                waveform=media.audio_waveform,
                log_mel=media.log_mel,
                sample_rate=media.sample_rate,
                has_faces=media.has_faces,
            )
            sync_active = bool(sync_out.get("active"))
            if sync_active:
                z_s = float(sync_out["mismatch_logit"])
        except Exception as exc:
            logger.exception("Sync inference failed")
            sync_out = {"active": False, "error": str(exc)}

    _tick(0.78, "5/6  Quality-weighted fusion + temperature calibration…")
    fusion: FusionResult = engines["fusion"].fuse(
        z_visual=z_v,
        z_audio=z_a,
        z_sync=z_s,
        faces=media.faces,
        waveform=media.audio_waveform,
        sample_rate=media.sample_rate,
        sync_active=sync_active,
        sensitivity=sensitivity,
    )

    _tick(0.90, "6/6  Grad-CAM++ / spectrogram explainability…")
    try:
        bundle = engines["explainer"].explain(
            faces=media.faces,
            log_mel=media.log_mel,
            fusion=fusion,
            visual_out=visual_out,
            audio_out=audio_out,
            sync_out=sync_out,
        )
    except Exception as exc:
        logger.exception("Explainability failed")
        bundle = ExplanationBundle(
            heatmap_rgb=None,
            overlay_rgb=None,
            original_face=media.faces[0] if media.faces else None,
            spectrogram_overlay=None,
            spectrogram_raw=None,
            narrative=f"Explainability unavailable: {exc}\n\nFused P(synthetic)={fusion.probability:.4f}, label={fusion.label}",
            cam_available=False,
            metadata={"error": str(exc)},
        )
    _tick(1.0, "Pipeline complete.")
    try:
        ingestor.close()
    except Exception:
        pass
    return media, visual_out, audio_out, sync_out, fusion, bundle


def _show_image(arr: Optional[np.ndarray], caption: str) -> None:
    if arr is None:
        st.info(f"{caption}: not available for this modality.")
        return
    img = np.asarray(arr)
    if img.dtype != np.uint8:
        if img.max() <= 1.5:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    show_image(img, caption)


def render_sidebar() -> Tuple[Optional[Any], float, str]:
    st.sidebar.markdown("## Neural Storm FT-02")
    st.sidebar.caption("Deepfake & Synthetic Media Detection Engine")
    uploaded = st.sidebar.file_uploader(
        "Evidence file",
        type=list(config.SUPPORTED_UPLOAD_TYPES),
        help="MP4, AVI, MOV, JPG, PNG, WAV, MP3",
    )
    sensitivity = st.sidebar.slider(
        "Decision sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Higher values shrink the uncertain band (more decisive Authentic / Synthetic labels).",
    )
    if uploaded is None and st.session_state.get("ns_demo"):
        st.sidebar.info("Demo mode: a built-in synthetic voice clip is loaded.")
        if st.sidebar.button("Exit demo", key="exit_demo"):
            st.session_state["ns_demo"] = False
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    st.sidebar.markdown(
        "<div class='ns-muted'><b>Accepted inputs</b><br/>🖼 Image · JPG, PNG<br/>🎞 Video · MP4, AVI, MOV<br/>🎙 Audio · WAV, MP3<br/>"
        "📷 Camera snapshot · 🎤 Microphone clip (Camera tab)</div>", unsafe_allow_html=True)
    device_choice = st.sidebar.radio(
        "Execution device",
        options=["auto", "cuda", "cpu"],
        index=0,
        help="Auto selects CUDA when a GPU is visible, otherwise CPU.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div class='ns-muted'>Models dir: <code>{config.MODELS_DIR}</code><br/>"
        f"Temp dir: <code>{config.TEMP_DIR}</code><br/>"
        f"Default weights w_v={config.W_VISUAL:.2f}, w_a={config.W_AUDIO:.2f}, "
        f"w_s={config.W_SYNC:.2f} · T={config.TEMPERATURE}</div>",
        unsafe_allow_html=True,
    )
    return uploaded, sensitivity, device_choice



def _media_kind(name: str, mime: str) -> str:
    """Route by extension first (browsers often send an empty or odd MIME type)."""
    kind = config.MEDIA_KIND_MAP.get(Path(name).suffix.lower())
    if kind:
        return kind
    for k in ("image", "audio", "video"):
        if (mime or "").startswith(k + "/"):
            return k
    return "video"


def _landing_cards() -> None:
    st.markdown("### Start an investigation")
    st.markdown("Upload a media sample from the left panel, or open the **Camera · Mic · Hand Lab** tab for a live snapshot or voice clip. "
                "Everything below runs on the evidence you provide.")
    cards = [
        ("01 · VISUAL", "Image forensics", "Sharpness, noise, FFT spectrum, ELA, metadata and provenance markers."),
        ("02 · HANDS", "Hand analytics", "21-point skeleton, joint angles, gestures and a plausibility score."),
        ("03 · AUDIO", "Voice signals", "Spectral patterns and audio spoof indicators when audio exists."),
        ("04 · FUSION", "Evidence fusion", "Calibrated, uncertainty-aware verdict across the available streams."),
    ]
    for col, (eyebrow, title, text) in zip(st.columns(4), cards):
        col.markdown(f"<div class='ns-card'><div class='ns-eyebrow'>{eyebrow}</div><h3>{title}</h3><p class='ns-muted'>{text}</p></div>",
                     unsafe_allow_html=True)
    st.markdown("#### What you can submit")
    kinds = [("🖼 Image", "JPG · PNG", "pixels, metadata, hands, faces"), ("🎞 Video", "MP4 · AVI · MOV", "frames + audio track + temporal checks"),
             ("🎙 Audio", "WAV · MP3", "signal lab + voice-spoof model"), ("📷 Camera / 🎤 Mic", "live snapshot or clip", "Camera & Hand Lab tab")]
    for col, (t, f, d) in zip(st.columns(4), kinds):
        col.markdown(f"<div class='ns-card'><b>{t}</b><br/><span class='ns-muted'>{f}<br/>{d}</span></div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**No file handy?** Try the built-in demo: a synthetic voice-like clip with deliberately detectable traits "
                "(band-limit, steady pitch, a digital-silence gap).")
    if st.button("▶ Load demo audio", key="load_demo"):
        st.session_state["ns_demo"] = True
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    st.caption("Research prototype · outputs are model-assisted indicators, not proof of authenticity or deception.")


def _render_media_summary(col_media: Any, media: Any, visual_out: Dict[str, Any], audio_out: Optional[Dict[str, Any]],
                          sync_out: Dict[str, Any], fusion: Any) -> None:
    """Fills the left column under the preview: what was analysed and why each stream did / did not run."""
    n_real_faces = len(getattr(media, "face_boxes", []) or [])
    has_audio = bool(getattr(media, "has_audio", False))
    with col_media:
        st.markdown("#### What was analysed")
        a, b, c = st.columns(3)
        a.metric("Frames", len(media.frames))
        b.metric("Faces found", n_real_faces)
        c.metric("Audio track", "yes" if has_audio else "none")

        probs = fusion.stream_probs or {}
        audio_ok = bool(audio_out and audio_out.get("available"))
        rows = [
            {"stream": "Visual (EfficientNet + DCT/SRM)", "ran": "yes" if visual_out and "error" not in visual_out else "no",
             "P(synthetic)": f"{probs['visual'] * 100:.0f}%" if "visual" in probs else "-",
             "why": "ran on a centre crop (no face detected)" if n_real_faces == 0 else f"ran on {n_real_faces} face crop(s)"},
            {"stream": "Audio (WavLM / AASIST)", "ran": "yes" if audio_ok else "no",
             "P(synthetic)": f"{probs['audio'] * 100:.0f}%" if "audio" in probs else "-",
             "why": "analysed first 12 s" if audio_ok else "video has no audio track" if not has_audio else "audio could not be analysed"},
            {"stream": "Lip-sync (SyncNet)", "ran": "yes" if sync_out.get("active") else "no",
             "P(synthetic)": f"{probs['sync'] * 100:.0f}%" if "sync" in probs else "-",
             "why": "ran" if sync_out.get("active") else "needs both a face and speech"},
        ]
        ui_labs.show_table(rows)
        if n_real_faces == 0 and media.media_type != "audio":
            st.info("No face was detected, so the visual score comes from a centre crop of the frame. "
                    "For screen recordings, slides, maps or other non-face content treat this verdict as not applicable.")
        active = sum(r["ran"] == "yes" for r in rows)
        st.caption(f"{active} of 3 streams contributed. Weights are renormalised over the streams that ran, "
                   "which is why w_v can be 1.00 when audio and lip-sync are unavailable.")


def _render_verdict_and_evidence(media: Any, visual_out: Dict[str, Any], audio_out: Optional[Dict[str, Any]],
                                 sync_out: Dict[str, Any], fusion: Any, bundle: Any, col_metrics: Any) -> None:
    with col_metrics:
        st.subheader("Calibrated verdict")
        if go is not None:
            show_plot(_gauge(fusion.probability_pct, fusion.label))
        else:
            st.metric("P(synthetic)", f"{fusion.probability_pct:.1f}%")
        st.markdown(f"<span class='ns-badge {_badge_class(fusion.label)}'>{fusion.label}</span>", unsafe_allow_html=True)
        lo, hi = fusion.confidence_interval
        st.markdown(
            f"<p class='ns-muted'>P(synthetic) = <b>{fusion.probability_pct:.1f}%</b> "
            f"· 95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]<br/>"
            f"Epistemic uncertainty = {fusion.epistemic_uncertainty:.3f} "
            f"· fused logit = {fusion.fused_logit:+.3f}</p>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("w_v", f"{fusion.weights['w_v']:.2f}")
        m2.metric("w_a", f"{fusion.weights['w_a']:.2f}")
        m3.metric("w_s", f"{fusion.weights['w_s']:.2f}")
        if visual_out.get("using_fallback_weights", True):
            st.caption("⚠️ No trained checkpoints in models/: the verdict is a heuristic demo signal, not a measured accuracy.")

    st.markdown("---")
    st.markdown("## Evidence lab")
    st.markdown("<p class='ns-muted'>Inspect the visual, explainability and audio evidence that contributed to the current result.</p>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        _show_image(bundle.original_face, "Original facial ROI (224×224)")
    with d2:
        _show_image(bundle.overlay_rgb if bundle.overlay_rgb is not None else bundle.heatmap_rgb, "Grad-CAM++ overlay")
    with d3:
        _show_image(bundle.spectrogram_overlay if bundle.spectrogram_overlay is not None else bundle.spectrogram_raw,
                    "Spectrogram anomaly overlay")

    st.markdown("#### Logit-grounded narrative")
    st.code(bundle.narrative, language="text")

    with st.expander("Stream logits, SQI, and ingestion metadata"):
        st.json(
            {
                "media_type": media.media_type,
                "mime_type": media.mime_type,
                "n_faces": len(media.faces),
                "n_frames": len(media.frames),
                "has_audio": media.has_audio,
                "visual": {k: v for k, v in visual_out.items() if k not in {"sequence_features", "spatial_features", "freq_features", "pooled_features"}},
                "audio": None if audio_out is None else {k: v for k, v in audio_out.items() if k not in {"pooled_features", "features"}},
                "sync": sync_out,
                "fusion": {
                    "fused_logit": fusion.fused_logit, "probability": fusion.probability, "label": fusion.label,
                    "weights": fusion.weights, "sqi": fusion.sqi, "stream_logits": fusion.stream_logits,
                    "stream_probs": fusion.stream_probs, "epistemic": fusion.epistemic_uncertainty,
                    "ci": fusion.confidence_interval, "metadata": fusion.metadata,
                },
            },
        )


def render_studio(uploaded: Optional[Any], sensitivity: float, device_choice: str) -> None:
    engines: Optional[Dict[str, Any]] = None
    if CORE_OK:
        try:
            engines = load_engines(device_choice)
        except Exception as exc:
            st.error(f"Engine initialization failed: {exc}")
            logger.exception("Engine bootstrap failed")
        if engines:
            chips = " · ".join(f"{k}: {v}" for k, v in engines.get("status", {}).items())
            st.caption(f"Runtime device **{engines.get('device', 'cpu')}**  ·  weight status  {chips or 'n/a'}")
    else:
        st.warning(f"The deep detection pipeline is unavailable ({CORE_ERR}). Install `requirements.txt` to enable it. "
                   "Image/hand/face metrics, camera lab, explorer and research tabs still work.")

    if uploaded is None:
        _landing_cards()
        return

    raw = uploaded.getvalue()
    kind = _media_kind(uploaded.name, uploaded.type or "")
    profile = ui_labs.profile_for(uploaded.name, raw, uploaded.type or "",
                                  "Built-in demo" if isinstance(uploaded, labs_demo.DemoUpload) else "File upload", CORE_OK)
    ui_labs.render_input_banner(profile)
    col_media, col_metrics = st.columns([1.15, 1.0])
    with col_media:
        st.subheader("Evidence preview")
        if kind == "image":
            show_image(raw, None)
        elif kind == "audio":
            st.audio(raw)
        else:
            st.video(raw)
            st.caption(f"Long videos are sampled: {config.MAX_FRAMES_CUDA if (engines and engines['device'] == 'cuda') else config.MAX_FRAMES_CPU} frames spread across the whole clip "
                       f"and the first {config.MAX_AUDIO_SECONDS:.0f} s of audio are analysed.")

    result = None
    if engines is not None:
        _sweep_temp()
        key = (hashlib.sha256(raw).hexdigest(), round(float(sensitivity), 2), engines["device"])
        cached = st.session_state.get("ns_pipeline")
        if cached and cached["key"] == key:
            result = cached["result"]
        else:
            progress = st.progress(0.0, text="Preparing analysis…")
            status_box = st.empty()
            path: Optional[Path] = None
            try:
                path = _save_upload(uploaded)
                result = run_pipeline(path, engines, sensitivity, progress, status_box)
                st.session_state["ns_pipeline"] = {"key": key, "result": result}
            except Exception as exc:
                progress.progress(1.0, text="Failed.")
                st.error(f"Pipeline failed: {exc}")
                st.code(traceback.format_exc())
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
                if result is not None:
                    job = result[0].metadata.get("job_dir")
                    if job:
                        shutil.rmtree(job, ignore_errors=True)
            status_box.empty()
            progress.empty()

    if result is not None:
        media, visual_out, audio_out, sync_out, fusion, bundle = result
        _render_verdict_and_evidence(media, visual_out, audio_out, sync_out, fusion, bundle, col_metrics)
        _render_media_summary(col_media, media, visual_out, audio_out, sync_out, fusion)
    else:
        media = None

    st.markdown("---")
    st.markdown("## 🧾 Input dossier")
    st.markdown("<p class='ns-muted'>Exactly what was submitted: file facts, media details, audio stream and which analyses apply.</p>", unsafe_allow_html=True)
    ui_labs.render_dossier(profile, key="up")
    if kind in ("audio", "video"):
        st.markdown("---")
        ui_labs.render_audio_lab(profile, key="up")

    st.markdown("---")
    if kind == "image":
        st.markdown("## Image & hand lab")
        from PIL import Image

        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        ui_labs.render_image_lab(rgb, raw, uploaded.name, key="up", profile=profile)
    elif kind == "video":
        st.markdown("## Video frame lab")
        frames = media.frames if media is not None else _quick_frames(raw, uploaded.name)
        ui_labs.render_video_lab(frames, uploaded.name, profile)


def _quick_frames(raw: bytes, name: str, n: int = 16) -> list:
    """Sample frames with OpenCV only (used when the deep pipeline is unavailable)."""
    import cv2

    fd, tmp = tempfile.mkstemp(prefix="ns_upload_", suffix=Path(name).suffix, dir=str(config.TEMP_DIR))
    os.close(fd)
    try:
        Path(tmp).write_bytes(raw)
        cap = cv2.VideoCapture(tmp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        for idx in (np.linspace(0, max(total - 1, 0), n).astype(int) if total > 0 else range(0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    finally:
        Path(tmp).unlink(missing_ok=True)


def _safe(fn: Any, *args: Any) -> None:
    """Run one tab body; an error in it must not take the other tabs down."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tab failed")
        st.error(f"This section failed: {exc}")
        st.code(traceback.format_exc())


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    uploaded, sensitivity, device_choice = render_sidebar()
    if uploaded is None and st.session_state.get("ns_demo"):
        uploaded = labs_demo.DemoUpload("demo_voice_synthetic.wav", labs_demo.demo_voice_wav(), "audio/wav")

    st.markdown("""
    <section class="ns-hero">
      <div class="ns-eyebrow">MULTIMODAL MEDIA INTELLIGENCE · IMAGE · VIDEO · AUDIO · FT-02 · LABS v2.2</div>
      <h1 style="margin:.35rem 0 .4rem 0">Neural Storm <span style="color:#61f2c2">/</span> Forensic Studio</h1>
      <div class="ns-subtitle">Investigate suspicious media with visual artifact analysis, hand and face metrics, audio spoof signals, lip-sync consistency and explainable evidence — in one workspace. Every submission starts with a clear <b>input dossier</b>: what you gave us, and what we can (and cannot) analyse from it.</div>
    </section>
    """, unsafe_allow_html=True)

    t_studio, t_cam, t_cmp, t_explorer, t_research, t_log = st.tabs(
        ["🔬 Forensic Studio", "📷 Camera · Mic · Hand Lab", "🔀 Tamper Compare", "🗂 Project Explorer", "📚 Research", "🧾 Session log & custody"])
    with t_studio:
        _safe(render_studio, uploaded, sensitivity, device_choice)
    with t_cam:
        _safe(ui_labs.render_camera_tab)
    with t_cmp:
        _safe(ui_labs.render_compare_tab)
    with t_explorer:
        _safe(ui_labs.render_explorer_tab)
    with t_research:
        _safe(ui_labs.render_research_tab)
    with t_log:
        _safe(ui_labs.render_log_tab)


if __name__ == "__main__":
    main()
