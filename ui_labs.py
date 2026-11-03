"""Streamlit rendering for Neural Storm Labs (image metrics, hands, liveness, explorer, research, log).

Only this module and app.py import streamlit; all logic lives in the torch-free ``labs`` package.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

import config
from labs import compare, custody, explorer, input_profile, liveness, report, triage
from labs import hand_analytics as ha
from labs import image_metrics as im
from labs.landmark_backend import LandmarkBackend, download_models, draw_detections
from labs.research_content import METRIC_GLOSSARY, TOPICS

ROOT = Path(__file__).resolve().parent
LEVEL_ICON = {"warn": "⚠️", "info": "ℹ️", "weak": "🔹"}
STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "ok": "✅", "missing": "❌", "info": "ℹ️"}


# ----------------------------------------------------------------------------- compat helpers
def _try(call_variants) -> None:
    """Call a Streamlit API with width keywords that differ between releases; use the first that works."""
    last: Optional[Exception] = None
    for fn in call_variants:
        try:
            fn()
            return
        except TypeError as exc:
            last = exc
        except Exception as exc:  # StreamlitAPIException for an unsupported width value
            last = exc
    if last:
        raise last


def show_image(arr: Any, caption: Optional[str] = None) -> None:
    _try([
        lambda: st.image(arr, caption=caption, use_container_width=True),
        lambda: st.image(arr, caption=caption, width="stretch"),
        lambda: st.image(arr, caption=caption, use_column_width=True),
        lambda: st.image(arr, caption=caption),
    ])


def show_plot(fig: Any) -> None:
    _try([
        lambda: st.plotly_chart(fig, use_container_width=True),
        lambda: st.plotly_chart(fig, width="stretch"),
        lambda: st.plotly_chart(fig),
    ])


def show_table(rows: Any, height: Optional[int] = None) -> None:
    kw = {"height": height} if height else {}
    _try([
        lambda: st.dataframe(rows, use_container_width=True, hide_index=True, **kw),
        lambda: st.dataframe(rows, width="stretch", hide_index=True, **kw),
        lambda: st.dataframe(rows, **kw),
    ])


def _kv_rows(d: Dict[str, Any], skip: tuple = ()) -> List[Dict[str, Any]]:
    rows = []
    for k, v in d.items():
        if k in skip or isinstance(v, dict) or (isinstance(v, list) and v and isinstance(v[0], dict)):
            continue
        if isinstance(v, float):
            v = round(v, 4)
        elif isinstance(v, (list, tuple)):
            v = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v)
        rows.append({"metric": k.replace("_", " "), "value": "-" if v is None else str(v)})
    return rows


def _flags(flags: List[Dict[str, str]]) -> None:
    for f in flags:
        st.markdown(f"{LEVEL_ICON.get(f['level'], '•')} {f['text']}")


# ----------------------------------------------------------------------------- engines
@st.cache_resource(show_spinner=False)
def get_backend() -> LandmarkBackend:
    return LandmarkBackend(config.MODELS_DIR)


def render_backend_status(backend: LandmarkBackend, key: str = "main") -> None:
    s = backend.status()
    st.caption(f"Landmark backend **{s['mode']}** · mediapipe {s['mediapipe_version'] or 'not installed'} · "
               f"hands {'ready' if s['hands_ready'] else 'NOT ready'} · faces via {s['face_model']}")
    if s["needs_model_download"] and s["mode"] == "tasks":
        st.info("This MediaPipe version needs two small model files for hand/face landmarks "
                f"(saved in `{s['models_dir']}`). Press the button, or run `python scripts/download_models.py`.")
        if st.button("Download MediaPipe models", key=f"dl_models_{key}"):
            with st.spinner("Downloading…"):
                for line in download_models(config.MODELS_DIR):
                    st.write(line)
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()


# ----------------------------------------------------------------------------- analysis bundle
def analyze_frame(rgb: np.ndarray, raw: Optional[bytes], name: str, backend: LandmarkBackend) -> Dict[str, Any]:
    """Run every image/hand/face analysis once for a frame. Cached in session_state by content hash."""
    sig = hashlib.sha256((raw or rgb.tobytes())).hexdigest() + "|" + backend.mode + str(backend.status()["hands_ready"])
    cache = st.session_state.setdefault("ns_lab_cache", {})
    if sig in cache:
        return cache[sig]

    det = backend.detect(rgb)
    h, w = rgb.shape[:2]
    hands = [ha.analyze_hand(hd.landmarks, w, h, hd.label, hd.score) for hd in det.hands]
    faces = []
    for fd in det.faces:
        faces.append({"box": list(fd.box), "score": fd.score, "source": fd.source,
                      "width_px": fd.box[2] - fd.box[0], "height_px": fd.box[3] - fd.box[1],
                      "consistency": im.face_region_consistency(rgb, fd.box)})
    image = im.analyze_image(rgb, raw, name)
    scene = ha.scene_checks(hands, len(faces))
    record = report.build_record(name, image, hands, faces, scene, backend.status(), det.notes)
    bundle = {
        "sig": sig, "name": name, "rgb": rgb, "annotated": draw_detections(rgb, det), "detection": det,
        "image": image, "hands": hands, "faces": faces, "scene": scene, "record": record,
    }
    if len(cache) >= 6:  # keep memory bounded
        cache.pop(next(iter(cache)))
    cache[sig] = bundle
    _log_history(record, sig)
    return bundle


def _log_history(record: Dict[str, Any], sig: str) -> None:
    seen = st.session_state.setdefault("ns_hist_seen", set())
    if sig in seen:
        return
    seen.add(sig)
    row = report.history_row(record)
    st.session_state.setdefault("ns_history", []).append({"kind": "image", **row})
    _chain_add("image", record["source_name"], sig.split("|")[0],
               f"{row['size']} px; {row['faces']} face(s), {row['hands']} hand(s); {row['flags']} warning(s)")


def _chain_add(kind: str, source: str, sha: str, summary: str) -> None:
    """Append one entry to the tamper-evident custody chain (once per kind+content)."""
    seen = st.session_state.setdefault("ns_chain_seen", set())
    if (kind, sha) in seen:
        return
    seen.add((kind, sha))
    custody.append(st.session_state.setdefault("ns_chain", []), kind, source, sha, summary)


def profile_for(name: str, raw: bytes, mime: str = "", source: str = "File upload", deep_ok: Optional[bool] = None) -> Dict[str, Any]:
    """Input dossier for a piece of evidence, cached per content and logged once."""
    key = hashlib.sha256(raw).hexdigest() + "|" + source
    cache = st.session_state.setdefault("ns_profiles", {})
    if key not in cache:
        if len(cache) >= 6:
            cache.pop(next(iter(cache)))
        cache[key] = input_profile.profile_upload(name, raw, mime, source, deep_ok)
        p = cache[key]
        if p["kind"] != "image":  # images are logged by the image lab
            a = (p.get("audio") or {})
            warns = sum(f["level"] == "warn" for f in a.get("flags", []))
            st.session_state.setdefault("ns_history", []).append(
                {"time_utc": p["received_utc"], "kind": p["kind"], "source": p["name"], "size": p["size_human"], "flags": warns,
                 "sha256_12": p["sha256"][:12]})
            _chain_add(p["kind"], p["name"], p["sha256"], f"{p['headline']}; {warns} audio warning(s)")
    return cache[key]


# ----------------------------------------------------------------------------- bundle UI
def render_bundle(b: Dict[str, Any], key: str, extra: Optional[Dict[str, Any]] = None) -> None:
    extra = extra or {}
    img_m = b["image"]["metrics"]
    hands, faces = b["hands"], b["faces"]
    n_warn = sum(f["level"] == "warn" for f in b["image"]["flags"])
    tri = triage.triage_score(image_flags=b["image"]["flags"], hands=hands if b["detection"].hands_available else None, faces=faces,
                              audio_flags=extra.get("audio_flags"), temporal_flags=extra.get("temporal_flags"))
    b["triage"] = tri

    c = st.columns(6)
    c[0].metric("Faces", len(faces))
    c[1].metric("Hands", len(hands))
    c[2].metric("Sharpness", f"{img_m['sharpness']['laplacian_variance']:.0f}")
    c[3].metric("Noise σ", "n/a" if img_m["noise"]["sigma"] != img_m["noise"]["sigma"] else f"{img_m['noise']['sigma']:.2f}")
    c[4].metric("ELA mean", f"{img_m['ela']['mean_error']:.2f}")
    c[5].metric("Warnings", n_warn)

    t_over, t_img, t_freq, t_face, t_hand, t_meta, t_export = st.tabs(
        ["Overview", "Image metrics", "Frequency · ELA · Noise", "Faces", "Hands", "Metadata", "Export"])

    with t_over:
        a, bcol = st.columns([1.5, 1])
        with a:
            show_image(b["annotated"], "Detections · mint box = face · coloured lines = hand skeleton")
        with bcol:
            st.markdown("**Indicators**")
            _flags(b["image"]["flags"])
            st.markdown("**Hands in scene**")
            _flags(b["scene"])
            for n in b["detection"].notes:
                st.caption(n)
        render_triage(tri)

    with t_img:
        x, y = st.columns(2)
        with x:
            st.markdown("**File & image**")
            show_table(_kv_rows(img_m["basic"]))
            st.markdown("**Perceptual hashes**")
            show_table(_kv_rows(img_m["hashes"]))
            st.markdown("**Sharpness · noise · grid**")
            show_table(_kv_rows({**img_m["sharpness"], **img_m["noise"]}))
        with y:
            st.markdown("**Exposure & colour**")
            show_table(_kv_rows(img_m["exposure"]))
            st.markdown("**Dominant colours**")
            sw = "".join(
                f"<span style='display:inline-block;width:{max(8, d['share_pct'] * 2.2):.0f}px;height:22px;background:{d['hex']};"
                f"border:1px solid rgba(255,255,255,.25)' title='{d['hex']} {d['share_pct']:.1f}%'></span>"
                for d in img_m["dominant_colors"])
            st.markdown(sw, unsafe_allow_html=True)
            show_table([{"colour": d["hex"], "share %": round(d["share_pct"], 1)} for d in img_m["dominant_colors"]])
        with st.expander("What do these metrics mean?"):
            show_table(METRIC_GLOSSARY)

    with t_freq:
        maps = b["image"]["maps"]
        p, q, r = st.columns(3)
        with p:
            show_image(maps["ela"], "ELA (re-save @ q90) · bright = higher error potential")
            show_table(_kv_rows(img_m["ela"]))
        with q:
            show_image(maps["spectrum"], "FFT log-magnitude · green dots = isolated peaks")
            show_table(_kv_rows(img_m["spectrum"], skip=("radial_profile_db",)))
        with r:
            show_image(maps["noise_residual"], "Noise residual (|image − median3|)")
            show_table(_kv_rows(img_m["noise_residual"]))
        prof = img_m["spectrum"].get("radial_profile_db")
        if prof:
            try:
                import plotly.graph_objects as go

                fig = go.Figure(go.Scatter(y=prof, mode="lines", line=dict(color="#61f2c2")))
                fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), title="Radial power spectrum (dB) · low→high frequency",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce7f5"})
                show_plot(fig)
            except Exception:
                st.line_chart(prof)
        st.caption("ELA and spectra are indicators for where to look; many benign operations (resizing, filters, re-saving) change them.")

    with t_face:
        if not faces:
            st.info("No face detected.")
        for i, f in enumerate(faces, 1):
            with st.expander(f"Face {i} · {f['width_px']}×{f['height_px']} px · {f['source']}", expanded=(i == 1)):
                x1, y1, x2, y2 = f["box"]
                crop = b["rgb"][y1:y2, x1:x2]
                a, bcol = st.columns([1, 2])
                with a:
                    if crop.size:
                        show_image(crop, "Face crop")
                with bcol:
                    st.metric("Detector confidence", "n/a (Haar)" if f["score"] is None else f"{f['score']:.2f}")
                    cons = f["consistency"]
                    if cons.get("available"):
                        show_table(_kv_rows({k: v for k, v in cons.items() if k not in ("available", "note")}))
                        st.caption(cons["note"])
                    else:
                        st.caption(f"Face-vs-surroundings check unavailable: {cons.get('reason')}")

    with t_hand:
        if not hands:
            st.info("No hands detected." + ("" if b["detection"].hands_available else " (Hand detection is not available - see the notice above.)"))
        for i, h in enumerate(hands, 1):
            pl = h["plausibility"]
            with st.expander(f"Hand {i} · {h['label']} · {h['gesture']} · {h['finger_count']} finger(s) · plausibility {pl['score']:.0f}/100",
                             expanded=(i == 1)):
                m = st.columns(5)
                m[0].metric("Gesture", h["gesture"])
                m[1].metric("Fingers up", h["finger_count"])
                m[2].metric("Openness", f"{h['openness_index']:.2f}")
                m[3].metric("Orientation", f"{h['orientation_deg']:.0f}°")
                m[4].metric("Confidence", "n/a" if h["detector_score"] is None else f"{h['detector_score']:.2f}")
                rows = [{"finger": n, "extended": d["extended"], "MCP°": round(d["mcp_angle"], 1), "PIP/IP°": round(d["pip_angle"], 1),
                         "DIP°": None if d["dip_angle"] != d["dip_angle"] else round(d["dip_angle"], 1),
                         "curl %": round(d["curl_pct"], 0), "length px": round(d["length_px"], 1), "len/middle": round(d["length_rel_middle"], 2)}
                        for n, d in h["fingers"].items()]
                show_table(rows)
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure(go.Bar(x=[r["finger"] for r in rows], y=[r["curl %"] for r in rows],
                                           marker_color=["#ff9f43", "#ffdd59", "#61f2c2", "#56ccf2", "#e978ff"]))
                    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="Finger curl %", yaxis_range=[0, 100],
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce7f5"})
                    show_plot(fig)
                except Exception:
                    pass
                st.markdown(f"**Skeleton plausibility: {pl['score']:.0f}/100 - {pl['label']}** ({pl['n_warn']} warn, {pl['n_fail']} fail)")
                show_table([{"ok?": STATUS_ICON[c['status']], "check": c["check"], "value": c["value"], "expected": c["expected"]} for c in pl["checks"]])
                st.markdown("**Geometry**")
                show_table(_kv_rows({k: v for k, v in h.items() if k not in ("extended", "fingers", "plausibility", "spread_deg", "bbox_px", "label", "gesture")}))
                show_table(_kv_rows(h["spread_deg"]))
        st.caption("Landmark models regularise hand shapes, so these checks flag skeleton-level oddities and low confidence - they cannot see every extra or merged finger.")

    with t_meta:
        meta = img_m["metadata"]
        show_table(_kv_rows({k: v for k, v in meta.items() if k != "exif"}))
        if meta.get("exif"):
            show_table([{"tag": k, "value": v} for k, v in meta["exif"].items()], height=300)
        else:
            st.caption("No EXIF tags.")

    with t_export:
        rec = {**b["record"], "input_dossier": input_profile.export_profile(b["profile"]) if b.get("profile") else None, "triage": tri}
        stem = Path(b["name"]).stem or "image"
        imgs = {"Annotated detections": b["annotated"], "ELA": b["image"]["maps"]["ela"],
                "FFT spectrum": b["image"]["maps"]["spectrum"], "Noise residual": b["image"]["maps"]["noise_residual"]}
        st.download_button("Download JSON metrics", json.dumps(rec, indent=2), f"{stem}_neural_storm_metrics.json", "application/json", key=f"{key}_json")
        st.download_button("Download HTML evidence report", report.build_html_report(rec, imgs), f"{stem}_neural_storm_report.html", "text/html", key=f"{key}_html")
        buf = io.BytesIO()
        from PIL import Image

        Image.fromarray(b["annotated"]).save(buf, format="PNG")
        st.download_button("Download annotated image (PNG)", buf.getvalue(), f"{stem}_annotated.png", "image/png", key=f"{key}_png")
        st.caption(report.DISCLAIMER)


def render_image_lab(rgb: np.ndarray, raw: Optional[bytes], name: str, key: str, profile: Optional[Dict[str, Any]] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    backend = get_backend()
    render_backend_status(backend, key=key)
    with st.spinner("Computing image, face and hand metrics…"):
        bundle = analyze_frame(rgb, raw, name, backend)
    bundle["profile"] = profile
    render_bundle(bundle, key, extra)
    return bundle


def render_video_lab(frames: List[np.ndarray], name: str, profile: Optional[Dict[str, Any]] = None) -> None:
    """Frame-by-frame lab for video: pick a frame for the full image lab, plus a temporal consistency chart."""
    if not frames:
        st.info("No frames were extracted.")
        return
    series = im.temporal_series(frames)
    aud = (profile or {}).get("audio")
    extra = {"temporal_flags": series.get("flags") if series.get("available") else None,
             "audio_flags": aud["flags"] if aud and aud.get("available") else None}
    if series.get("available"):
        st.markdown("**Temporal consistency**")
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_scatter(y=series["sharpness"], name="sharpness (Laplacian var)")
            fig.add_scatter(y=series["noise_sigma"], name="noise σ", yaxis="y2")
            fig.add_scatter(y=series["frame_diff"], name="frame difference", yaxis="y2")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), yaxis2=dict(overlaying="y", side="right"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce7f5"}, legend=dict(orientation="h"))
            show_plot(fig)
        except Exception:
            st.line_chart({"sharpness": series["sharpness"], "noise": series["noise_sigma"]})
        show_table(_kv_rows({k: v for k, v in series.items() if not isinstance(v, list) and k != "available"}))
        for f in series["flags"]:
            st.markdown(f"⚠️ {f}")
        if not series["flags"]:
            st.caption("No strong frame-to-frame flicker or splice indicators.")
    idx = st.slider("Frame for detailed lab", 0, len(frames) - 1, len(frames) // 2, key=f"vid_frame_{name}")
    render_image_lab(np.asarray(frames[idx]), None, f"{name} [frame {idx}]", key=f"vid_{idx}", profile=profile, extra=extra)


# ----------------------------------------------------------------------------- camera tab
def render_camera_tab() -> None:
    st.markdown("### Camera & microphone lab · face, hand and voice metrics")
    st.caption("Take a snapshot in your browser. Faces and hands are located and measured; nothing is identified. The image is processed in memory only.")
    backend = get_backend()
    frame = st.camera_input("Capture a frame", key="neural_storm_camera")

    bundle = None
    if frame is not None:
        try:
            from PIL import Image

            raw = frame.getvalue()
            rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            bundle = analyze_frame(rgb, raw, "camera_frame.png", backend)
            bundle["profile"] = profile_for("camera_frame.png", raw, "image/png", "Camera snapshot")
            render_input_banner(bundle["profile"])
        except Exception as exc:
            st.error(f"Camera analysis failed: {exc}")

    render_liveness(bundle, frame.getvalue() if frame is not None else None)

    render_backend_status(backend, key="cam")
    if bundle is not None:
        render_bundle(bundle, key="cam")
    else:
        st.info("Take a snapshot (show a face and/or a hand) to see detections and all metrics.")

    st.markdown("---")
    st.markdown("#### 🎙 Microphone clip · voice metrics")
    st.caption("Record a few seconds of speech. The clip is measured in memory only (loudness, clipping, bandwidth, pitch, digital-silence gaps…).")
    mic = st.audio_input("Record a voice clip", key="neural_storm_mic") if hasattr(st, "audio_input") else None
    if not hasattr(st, "audio_input"):
        st.info("Microphone recording needs Streamlit 1.39 or newer (`pip install -U streamlit`). You can still upload audio files in the sidebar.")
    if mic is not None:
        mp = profile_for("mic_recording.wav", mic.getvalue(), "audio/wav", "Microphone recording")
        render_input_banner(mp)
        render_audio_lab(mp, key="mic")


def render_liveness(bundle: Optional[Dict[str, Any]], raw: Optional[bytes]) -> None:
    with st.expander("🎲 Gesture challenge (experimental liveness check)", expanded=False):
        st.caption("Random, time-limited instruction verified from your snapshot. Research prototype: it does not stop real-time deepfakes or prepared images.")
        ch = st.session_state.get("ns_challenge")
        if st.button("New challenge", key="new_challenge"):
            ch = liveness.new_challenge()
            st.session_state["ns_challenge"] = ch
            st.session_state.pop("ns_challenge_result", None)
        if ch is None:
            st.info("Press **New challenge**, follow the instruction, take a snapshot, then press **Verify snapshot**.")
            return
        remaining = max(0, int(ch["ttl"] - (time.time() - ch["issued_at"])))
        st.markdown(f"**Challenge `{ch['id']}`:** {ch['prompt']}  \n⏱ about {remaining}s remaining when this page last refreshed.")
        if st.button("Verify snapshot", key="verify_challenge"):
            if bundle is None or raw is None:
                st.warning("Take a camera snapshot first.")
            else:
                used = st.session_state.setdefault("ns_used_frames", set())
                sha = liveness.frame_hash(raw)
                result = liveness.verify(ch, bundle["hands"], sha, used)
                if result["passed"]:
                    used.add(sha)
                st.session_state["ns_challenge_result"] = result
        res = st.session_state.get("ns_challenge_result")
        if res:
            (st.success if res["passed"] else st.error)(("PASSED - " if res["passed"] else "NOT PASSED - ") + res["reason"])


# ----------------------------------------------------------------------------- explorer / research / log
def render_explorer_tab() -> None:
    st.markdown("### Project explorer")
    rows = explorer.scan_project(ROOT)
    total_lines = sum(r["lines"] or 0 for r in rows if r["path"].endswith(".py"))
    c = st.columns(3)
    c[0].metric("Files", len(rows))
    c[1].metric("Python lines", f"{total_lines:,}")
    c[2].metric("Total size", f"{sum(r['size_bytes'] for r in rows) / 1024:.0f} KB")
    show_table([{"path": r["path"], "type": r["type"], "size (B)": r["size_bytes"], "lines": r["lines"], "role": r["description"]} for r in rows])

    text_files = [r["path"] for r in rows if r["text"]]
    pick = st.selectbox("Open a file", text_files, index=text_files.index("app.py") if "app.py" in text_files else 0, key="explorer_pick")
    try:
        body = explorer.read_text(ROOT, pick)
        lang = "python" if pick.endswith(".py") else "markdown" if pick.endswith(".md") else "text"
        st.code(body, language=lang, line_numbers=True) if _supports_line_numbers() else st.code(body, language=lang)
    except Exception as exc:
        st.error(f"Cannot read file: {exc}")

    st.markdown("### Environment health check")
    health = explorer.health_check(ROOT)
    show_table([{"ok?": STATUS_ICON.get(h["status"], "•"), "item": h["item"], "status": h["status"], "detail": h["detail"]} for h in health])
    bad = [h for h in health if h["status"] == "missing"]
    if bad:
        st.warning(f"{len(bad)} item(s) missing. Install with `pip install -r requirements.txt`. The Labs tabs keep working without PyTorch; the Forensic Studio pipeline needs it.")


def _supports_line_numbers() -> bool:
    try:
        import inspect

        return "line_numbers" in inspect.signature(st.code).parameters
    except Exception:
        return False


def render_research_tab() -> None:
    st.markdown("### Research notes")
    st.caption("Gathered 18 Oct 2026 from public papers and guides; summaries are paraphrased - follow the links for exact claims.")
    for t in TOPICS:
        with st.expander(str(t["title"]), expanded=False):
            for f in t["findings"]:  # type: ignore[union-attr]
                st.markdown(f"- {f}")
            st.markdown(f"**In this project:** {t['in_project']}")
            for title, url in t["refs"]:  # type: ignore[union-attr]
                st.markdown(f"[{title}]({url})")
    st.markdown("### Metric glossary")
    show_table(METRIC_GLOSSARY)
    st.caption(report.DISCLAIMER)


def render_log_tab() -> None:
    st.markdown("### Session log")
    hist = st.session_state.get("ns_history", [])
    if not hist:
        st.info("Analyses you run in this session will be listed here.")
        return
    show_table(hist)
    import csv

    fields: List[str] = []
    for r in hist:  # rows of different media kinds carry different columns
        fields += [k for k in r if k not in fields]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(hist)
    st.download_button("Download session log (CSV)", buf.getvalue(), "neural_storm_session_log.csv", "text/csv")

    chain = st.session_state.get("ns_chain", [])
    st.markdown("### 🔗 Chain of custody")
    st.caption("Every analysed file is sealed into a hash chain: each entry contains the SHA-256 of the one before it, so editing or removing an "
               "entry in an exported log is detectable. It shows the *file is internally consistent*; it is not a digital signature.")
    ok, bad, msg = custody.verify(chain)
    (st.success if ok else st.error)(("Chain intact - " if ok else "Chain BROKEN - ") + msg)
    st.markdown(f"Head hash: `{custody.head(chain)}`")
    show_table([{"#": e["index"], "time_utc": e["time_utc"], "kind": e["kind"], "source": e["source"], "file sha256 (12)": e["file_sha256"][:12],
                 "summary": e["summary"], "entry hash (12)": e["entry_hash"][:12]} for e in chain])
    st.download_button("Download custody chain (JSON)", custody.export_json(chain), "neural_storm_custody_chain.json", "application/json", key="dl_chain")
    up = st.file_uploader("Verify a previously exported custody chain (.json)", type=["json"], key="verify_chain_upload")
    if up is not None:
        good, vmsg = custody.verify_export(up.getvalue().decode("utf-8", "replace"))
        (st.success if good else st.error)(("Valid - " if good else "Invalid - ") + vmsg)

    if st.button("Clear log", key="clear_log"):
        for k in ("ns_history", "ns_chain"):
            st.session_state[k] = []
        st.session_state["ns_hist_seen"] = set()
        st.session_state["ns_chain_seen"] = set()
        st.session_state["ns_lab_cache"] = {}
        st.session_state["ns_profiles"] = {}
        st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()


# ----------------------------------------------------------------------------- input dossier UI
def render_input_banner(p: Dict[str, Any]) -> None:
    """Prominent, unmistakable statement of what was provided and which analyses it enables."""
    pills = "".join(
        f"<span class='ns-pill{' ns-pill-on' if k == p['kind'] else ''}'>{label}{' · RECEIVED' if k == p['kind'] else ''}</span>"
        for k, label in (("image", "🖼 Image"), ("video", "🎞 Video"), ("audio", "🎙 Audio")))
    chips = "".join(
        f"<span class='ns-chip {'ns-chip-on' if r['applies'] else 'ns-chip-off'}' title='{html.escape(r['why'], quote=True)}'>"
        f"{'✓' if r['applies'] else '✕'} {html.escape(r['stream'])}</span>" for r in p["plan"])
    st.markdown(
        f"<div class='ns-input'><div class='ns-eyebrow'>INPUT RECEIVED · {html.escape(p['source'].upper())}</div>"
        f"<div class='ns-pills'>{pills}</div>"
        f"<div class='ns-input-title'>{html.escape(p['headline'])}</div>"
        f"<div class='ns-muted'>{html.escape(input_profile.plain_summary(p))}</div>"
        f"<div class='ns-muted' style='margin-top:.55rem'><b>What will be analysed</b> (hover a chip for the reason)</div>"
        f"<div>{chips}</div></div>", unsafe_allow_html=True)
    for n in p.get("notes", []):
        st.caption(n)
    if p["kind"] == "audio" and not (p.get("audio") or {}).get("available"):
        st.warning(f"Audio could not be measured: {p.get('audio_error') or (p.get('audio') or {}).get('reason') or 'unknown reason'}.")
    if p["kind"] == "video" and p["media"].get("has_audio_stream") is False:
        st.info("This video has **no audio track**, so audio forensics and lip-sync are skipped (the plan above shows why).")


def render_dossier(p: Dict[str, Any], key: str) -> None:
    """Full input dossier: file facts, media facts, audio stream, analysis plan, JSON export."""
    rows = input_profile.profile_rows(p)
    t1, t2, t3 = st.tabs(["File & media", "Audio stream", "Analysis plan"])
    with t1:
        a, b = st.columns(2)
        with a:
            st.markdown("**File**")
            show_table(rows["file"])
        with b:
            st.markdown("**Media details**")
            show_table(rows["media"] or [{"field": "-", "value": "no media details available"}])
    with t2:
        if rows["audio_stream"]:
            show_table(rows["audio_stream"])
        elif p["kind"] == "image":
            st.info("Images carry no audio stream.")
        else:
            st.info("No audio stream details available (the file has none, or ffprobe is not installed).")
    with t3:
        show_table(rows["plan"])
    st.download_button("Download input dossier (JSON)", json.dumps(input_profile.export_profile(p), indent=2),
                       f"{Path(p['name']).stem}_input_dossier.json", "application/json", key=f"{key}_dossier_json")


def render_triage(t: Dict[str, Any]) -> None:
    colour = {"High": "#ff9eae", "Medium": "#ffda83", "Low": "#7df6c8"}[t["label"]]
    st.markdown(
        f"<div class='ns-card'><div class='ns-eyebrow'>REVIEW-PRIORITY TRIAGE</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{colour}'>{t['score']:.0f}<span style='font-size:1rem;color:#a8b4c7'> / 100 · {t['label']}</span></div>"
        f"<div class='ns-bar'><div style='width:{min(100, t['score']):.0f}%;background:{colour}'></div></div>"
        f"<p class='ns-muted'>{html.escape(t['advice'])}</p></div>", unsafe_allow_html=True)
    if t["breakdown"]:
        show_table([{"signal": r["signal"], "points": r["points"], "why": r["why"]} for r in t["breakdown"]])
    st.caption("Measured: " + (", ".join(t["coverage"]) or "nothing") + (" · not measured: " + ", ".join(t["not_measured"]) if t["not_measured"] else "")
               + ". " + t["disclaimer"])


# ----------------------------------------------------------------------------- audio lab
def _spectrogram_image(v: Dict[str, Any]) -> np.ndarray:
    S = np.asarray(v["spec_db"], dtype=np.float32)
    n = ((S - S.min()) / (S.max() - S.min() + 1e-9) * 255).astype(np.uint8)
    img = cv2.resize(np.flipud(n), (960, 340), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(cv2.applyColorMap(img, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    fmax = float(v["spec_freqs"][-1])
    step = 1000 if fmax <= 12000 else 4000
    for hz in range(0, int(fmax) + 1, step):
        y = int(339 - hz / fmax * 339)
        cv2.line(rgb, (0, y), (12, y), (255, 255, 255), 1)
        cv2.putText(rgb, f"{hz // 1000}k" if hz else "0", (16, min(max(y + 4, 12), 336)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def _line_plot(x: Any, y: Any, title: str, xlab: str, ylab: str, colour: str = "#61f2c2") -> None:
    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=list(map(float, x)), y=list(map(float, y)), mode="lines", line=dict(color=colour, width=1.4)))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=36, b=10), title=title, xaxis_title=xlab, yaxis_title=ylab,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#dce7f5"})
        show_plot(fig)
    except Exception:
        st.caption(title)
        st.line_chart(list(map(float, y)))


def render_audio_lab(p: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """Signal-level audio forensics for an audio file, the audio track of a video, or a microphone clip."""
    a = p.get("audio")
    st.markdown("## 🎙 Audio lab")
    if not a or not a.get("available"):
        st.info("No audio could be measured: " + str((a or {}).get("reason") or p.get("audio_error") or "this file has no audio."))
        return None
    m, v = a["metrics"], a["viz"]
    dec = p.get("audio_decode", {})
    where = "the audio track of the video" if p["kind"] == "video" else "the recording"
    note = f" · {dec['note']}" if dec.get("note") else ""
    st.caption(f"Analysed the first {m['duration_s']:.1f} s of {where} · decoded to mono at {m['sample_rate_hz']:,} Hz "
               f"via {dec.get('decoder', 'n/a')} · {m['samples_analysed']:,} samples{note}")
    c = st.columns(6)
    c[0].metric("Duration", f"{m['duration_s']:.1f} s")
    c[1].metric("Sample rate", f"{m['sample_rate_hz'] / 1000:g} kHz")
    c[2].metric("Loudness (RMS)", f"{m['rms_dbfs']:.1f} dBFS")
    c[3].metric("Peak", f"{m['peak_dbfs']:.1f} dBFS")
    c[4].metric("Clipping", f"{m['clipping_pct']:.2f}%")
    c[5].metric("Near-silent", f"{m['silence_pct']:.0f}%")
    d = st.columns(6)
    d[0].metric("Median pitch", "n/a" if m["f0_median_hz"] is None else f"{m['f0_median_hz']:.0f} Hz")
    d[1].metric("Pitch spread σ", "n/a" if m["f0_std_hz"] is None else f"{m['f0_std_hz']:.1f} Hz")
    d[2].metric("Voiced frames", f"{m['voiced_ratio'] * 100:.0f}%")
    d[3].metric("Content up to", f"{m['bandwidth_cutoff_hz'] / 1000:.1f} kHz")
    d[4].metric("Noise floor", "n/a" if m["noise_floor_dbfs"] is None else f"{m['noise_floor_dbfs']:.0f} dBFS")
    d[5].metric("Digital-silence gaps", m["digital_silence_runs"])

    _line_plot(v["envelope_times"], v["envelope"], "Waveform envelope", "seconds", "amplitude")
    show_image(_spectrogram_image(v), "Spectrogram · time → · frequency ↑ (bright = more energy)")
    x1, x2 = st.columns(2)
    with x1:
        _line_plot(v["spec_freqs"], v["psd_db"], "Long-term spectrum (where the energy lives)", "Hz", "dB", "#a78bfa")
    with x2:
        _line_plot(np.arange(len(v["frame_rms_db"])) * 0.02, v["frame_rms_db"], "Loudness per 20 ms frame", "seconds", "dBFS", "#ffc86b")

    st.markdown("**Audio indicators**")
    _flags(a["flags"])
    tri = triage.triage_score(audio_flags=a["flags"])
    render_triage(tri)
    with st.expander("All audio measurements"):
        show_table(_kv_rows(m))
    with st.expander("What do these audio metrics mean?"):
        show_table(AUDIO_GLOSSARY)
    dossier = input_profile.export_profile(p)
    st.download_button("Download HTML input & audio report", report.build_input_report(dossier, tri), f"{Path(p['name']).stem}_audio_report.html",
                       "text/html", key=f"{key}_audio_html")
    st.caption(report.DISCLAIMER + " Audio indicators describe the signal; they cannot tell a synthetic voice from a clean recording on their own.")
    return tri


AUDIO_GLOSSARY = [
    {"metric": "dBFS", "meaning": "Level relative to digital full scale (0 = loudest possible). Speech RMS is usually −30 to −15 dBFS."},
    {"metric": "Clipping", "meaning": "Share of samples at the digital ceiling. High values mean distortion or heavy limiting."},
    {"metric": "Content up to (bandwidth)", "meaning": "Highest frequency with real energy. A hard stop well below Nyquist suggests a low-bitrate codec, phone line or up-sampling."},
    {"metric": "Noise floor", "meaning": "Level of the quietest frames. Real rooms are rarely below −70 dBFS; generated or gated audio can be."},
    {"metric": "Digital-silence gaps", "meaning": "Stretches of exact zeros (>= 60 ms). Natural recordings contain tiny noise; exact zeros point to edits or synthesis."},
    {"metric": "Median pitch / spread", "meaning": "Fundamental frequency of voiced frames. Very small spread = monotone delivery or a synthetic or sustained source."},
    {"metric": "Spectral flatness", "meaning": "0 = pure tone, 1 = white noise. Speech sits in between."},
    {"metric": "Abrupt level jumps", "meaning": "Sudden >30 dB changes between neighbouring 20 ms frames: possible splices or aggressive gating."},
]


# ----------------------------------------------------------------------------- compare tab
def _as_rgb(upload: Any) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(upload.getvalue())).convert("RGB"))


def render_compare_tab() -> None:
    st.markdown("### 🔀 Tamper compare · original vs suspect")
    st.caption("Upload an original and a suspected edit or re-post of the same image. The lab shows what changed, where, and how similar they are "
               "(pixel difference, structural similarity, perceptual hashes). Files are processed in memory only.")
    a, b = st.columns(2)
    ua = a.file_uploader("Original image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cmp_a")
    ub = b.file_uploader("Suspect image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cmp_b")
    if ua is None or ub is None:
        st.info("Add both images to run the comparison.")
        return
    try:
        ra, rb = _as_rgb(ua), _as_rgb(ub)
    except Exception as exc:
        st.error(f"Could not read one of the images: {exc}")
        return
    sig = hashlib.sha256(ua.getvalue() + hashlib.sha256(ub.getvalue()).digest()).hexdigest()
    cache = st.session_state.setdefault("ns_cmp", {})
    if sig not in cache:
        cache.clear()
        cache[sig] = compare.compare_images(ra, rb)
        _chain_add("compare", f"{ua.name} vs {ub.name}", sig, f"SSIM {cache[sig]['ssim']}; {cache[sig]['changed_pixels_pct']}% pixels changed")
    r = cache[sig]
    c = st.columns(4)
    c[0].metric("Similarity (SSIM)", f"{r['ssim']:.3f}")
    c[1].metric("Pixels changed", f"{r['changed_pixels_pct']:.1f}%")
    c[2].metric("Mean difference", f"{r['mean_abs_diff']:.1f}")
    c[3].metric("Changed regions", len(r["regions"]))
    (st.success if r["ssim"] > 0.995 else st.warning)(r["verdict"])
    if r["second_image_resized"]:
        st.caption("The suspect image has a different size and was resized to the original for comparison.")
    x, y, z = st.columns(3)
    with x:
        show_image(r["overlay"], "Overlay · boxes mark the largest changes")
    with y:
        show_image(r["heatmap"], "Difference heat-map")
    with z:
        show_image(rb, "Suspect image as uploaded")
    if r["regions"]:
        show_table(r["regions"])
    show_table(_kv_rows({f"hash distance ({k})": v for k, v in r["hash_hamming"].items()}))
    st.caption("Small hash distances mean visually similar images. A local edit can leave hashes close while SSIM and the heat-map expose it. "
               + report.DISCLAIMER)
