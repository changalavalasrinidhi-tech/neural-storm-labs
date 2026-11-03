"""Session records, JSON export and a self-contained HTML evidence report (no external assets)."""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from labs import __version__
from labs.image_metrics import to_jsonable

DISCLAIMER = ("Research prototype. All values are model-assisted indicators, not proof of authenticity, "
              "manipulation or identity. Landmarks are geometric points only; no face or person recognition is performed.")


def build_record(name: str, image_result: Dict[str, Any], hands: Sequence[Dict[str, Any]], faces: Sequence[Dict[str, Any]],
                 scene_notes: Sequence[Dict[str, str]], backend_status: Dict[str, Any], detection_notes: Sequence[str],
                 dossier: Optional[Dict[str, Any]] = None, triage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return to_jsonable({
        "tool": f"Neural Storm Labs {__version__}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": name,
        "disclaimer": DISCLAIMER,
        "image": image_result["metrics"],
        "faces": list(faces),
        "hands": list(hands),
        "scene_notes": list(scene_notes),
        "detector": {"backend": backend_status, "notes": list(detection_notes)},
        "input_dossier": dossier,
        "triage": triage,
    })


def history_row(record: Dict[str, Any]) -> Dict[str, Any]:
    img, hands = record["image"], record["hands"]
    return {
        "time_utc": record["generated_utc"],
        "source": record["source_name"],
        "size": f"{img['basic']['width']}x{img['basic']['height']}",
        "sharpness": round(img["sharpness"]["laplacian_variance"], 1),
        "noise_sigma": round(img["noise"]["sigma"], 2) if img["noise"]["sigma"] is not None else None,
        "ela_mean": round(img["ela"]["mean_error"], 2),
        "faces": len(record["faces"]),
        "hands": len(hands),
        "gestures": ", ".join(h["gesture"] for h in hands) or "-",
        "min_plausibility": round(min((h["plausibility"]["score"] for h in hands), default=float("nan")), 0) if hands else None,
        "flags": sum(1 for f in img["flags"] if f["level"] == "warn"),
        "sha256_12": (img["basic"].get("sha256") or "")[:12],
    }


REPORT_CSS = """
:root{--bg:#fff;--fg:#14202e;--mut:#5b6b7f;--card:#f3f6fa;--line:#d5dde8;--warn:#b45309;--info:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--fg:#e8eef8;--mut:#9fb0c6;--card:#111a2b;--line:#26344a;--warn:#ffc86b;--info:#8db4ff}}
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}
h1,h2,h3,h4{letter-spacing:-.02em}h2{border-bottom:1px solid var(--line);padding-bottom:.3rem;margin-top:2rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:.9rem}td,th{border:1px solid var(--line);padding:.35rem .55rem;text-align:left;word-break:break-word}
th{background:var(--card)}.muted{color:var(--mut)}.note{background:var(--card);border:1px solid var(--line);padding:.7rem 1rem;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}figure{margin:0}img{max-width:100%;border-radius:8px;border:1px solid var(--line)}
figcaption{color:var(--mut);font-size:.85rem}.warn b{color:var(--warn)}.info b,.weak b{color:var(--info)}
"""


def _png_b64(arr: np.ndarray, max_side: int = 900) -> str:
    a = np.asarray(arr)
    h, w = a.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return html.escape(str(v))


def _table(rows: Sequence[Sequence[Any]], head: Optional[Sequence[str]] = None) -> str:
    out = ["<table>"]
    if head:
        out.append("<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _kv(d: Dict[str, Any], skip: Sequence[str] = ()) -> str:
    """Two-column table of scalar (and scalar-list) values; nested dicts / lists of dicts are skipped."""
    rows = []
    for k, v in d.items():
        if k in skip or isinstance(v, dict) or (isinstance(v, list) and v and isinstance(v[0], dict)):
            continue
        rows.append((k, v))
    return _table(rows, ["Metric", "Value"])


def build_html_report(record: Dict[str, Any], images: Dict[str, np.ndarray]) -> str:
    img = record["image"]
    parts: List[str] = []
    parts.append(f"<h1>Neural Storm Labs - evidence report</h1><p class='muted'>{html.escape(record['source_name'])} &middot; generated {record['generated_utc']} UTC &middot; {html.escape(record['tool'])}</p>")
    parts.append(f"<div class='note'>{html.escape(record['disclaimer'])}</div>")
    parts.append(_dossier_html(record.get("input_dossier")))
    parts.append(_triage_html(record.get("triage")))

    parts.append("<h2>Indicators</h2><ul>" + "".join(f"<li class='{f['level']}'><b>{f['level'].upper()}</b> {html.escape(f['text'])}</li>" for f in img["flags"]) + "</ul>")

    if images:
        parts.append("<h2>Visual evidence</h2><div class='grid'>")
        for title, arr in images.items():
            parts.append(f"<figure><img alt='{html.escape(title)}' src='data:image/png;base64,{_png_b64(arr)}'><figcaption>{html.escape(title)}</figcaption></figure>")
        parts.append("</div>")

    parts.append("<h2>File and image</h2>" + _kv(img["basic"]))
    parts.append("<h3>Perceptual hashes</h3>" + _kv(img["hashes"]))
    parts.append("<h2>Exposure and colour</h2>" + _kv(img["exposure"]))
    parts.append(_table([(c["hex"], c["share_pct"]) for c in img["dominant_colors"]], ["Dominant colour", "Share %"]))
    parts.append("<h2>Sharpness, noise, grid</h2>" + _kv(img["sharpness"]) + _kv(img["noise"]))
    parts.append("<h2>Frequency spectrum</h2>" + _kv(img["spectrum"], skip=("radial_profile_db",)))
    parts.append("<h2>Error Level Analysis</h2>" + _kv(img["ela"]))
    parts.append("<h2>Noise residual</h2>" + _kv(img["noise_residual"]))
    meta = img["metadata"]
    parts.append("<h2>Metadata and provenance</h2>" + _kv({k: v for k, v in meta.items() if k != "exif"}))
    if meta.get("exif"):
        parts.append("<h3>EXIF</h3>" + _table(list(meta["exif"].items()), ["Tag", "Value"]))

    parts.append(f"<h2>Faces ({len(record['faces'])})</h2>")
    for i, f in enumerate(record["faces"], 1):
        parts.append(f"<h3>Face {i}</h3>" + _kv({k: v for k, v in f.items() if k != "consistency"}))
        if f.get("consistency"):
            parts.append(_kv(f["consistency"]))

    parts.append(f"<h2>Hands ({len(record['hands'])})</h2>")
    for n in record["scene_notes"]:
        parts.append(f"<p class='{n['level']}'>{html.escape(n['text'])}</p>")
    for i, h in enumerate(record["hands"], 1):
        parts.append(f"<h3>Hand {i} - {html.escape(h['label'])} - {html.escape(h['gesture'])}</h3>")
        parts.append(_kv({k: v for k, v in h.items() if k not in ("extended", "fingers", "plausibility", "spread_deg", "bbox_px")}))
        parts.append("<h4>Spread angles (deg)</h4>" + _kv(h["spread_deg"]))
        parts.append(_table([(n, d["extended"], d["pip_angle"], d["dip_angle"], d["curl_pct"], d["length_px"], d["length_rel_middle"])
                             for n, d in h["fingers"].items()],
                            ["Finger", "Extended", "PIP/IP angle", "DIP angle", "Curl %", "Length px", "Length / middle"]))
        pl = h["plausibility"]
        parts.append(f"<h4>Skeleton plausibility: {pl['score']:.0f}/100 ({html.escape(pl['label'])})</h4>")
        parts.append(_table([(c["check"], c["value"], c["expected"], c["status"]) for c in pl["checks"]], ["Check", "Value", "Expected", "Status"]))

    parts.append("<h2>Detector</h2>" + _kv(record["detector"]["backend"]) + "".join(f"<p class='muted'>{html.escape(n)}</p>" for n in record["detector"]["notes"]))
    css = REPORT_CSS
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Neural Storm Labs report</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def _dossier_html(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return ""
    out = ["<h2>Input dossier - what was submitted</h2>"]
    out.append(f"<div class='note'><b>{html.escape(str(d.get('kind', '')).upper())}</b> &middot; {html.escape(str(d.get('headline', '')))}</div>")
    out.append(_kv({"source": d.get("source"), "file name": d.get("name"), "extension": d.get("extension"), "declared MIME": d.get("mime_declared"),
                    "size": d.get("size_human"), "SHA-256": d.get("sha256"), "received (UTC)": d.get("received_utc")}))
    if d.get("media"):
        out.append("<h3>Media details</h3>" + _kv(d["media"]))
    if d.get("audio_stream"):
        out.append("<h3>Audio stream</h3>" + _kv(d["audio_stream"]))
    if d.get("plan"):
        out.append("<h3>Analysis plan</h3>" + _table([(r["stream"], "yes" if r["applies"] else "no", r["why"]) for r in d["plan"]], ["Analysis", "Runs", "Why"]))
    return "".join(out)


def _triage_html(t: Optional[Dict[str, Any]]) -> str:
    if not t:
        return ""
    out = [f"<h2>Review-priority triage: {t['score']:.0f}/100 ({html.escape(t['label'])})</h2>", f"<p>{html.escape(t['advice'])}</p>"]
    if t.get("breakdown"):
        out.append(_table([(r["signal"], r["points"], r["why"]) for r in t["breakdown"]], ["Signal", "Points", "Why"]))
    out.append(f"<p class='muted'>Measured: {html.escape(', '.join(t.get('coverage') or ['nothing']))}. Not measured: {html.escape(', '.join(t.get('not_measured') or ['-']))}. {html.escape(t.get('disclaimer', ''))}</p>")
    return "".join(out)


def build_input_report(dossier: Dict[str, Any], triage: Optional[Dict[str, Any]] = None) -> str:
    """Self-contained HTML report for audio / video evidence (the image report covers pixels)."""
    parts: List[str] = [f"<h1>Neural Storm Labs - input &amp; audio report</h1><p class='muted'>{html.escape(str(dossier.get('name', '')))} &middot; generated "
                        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC &middot; Neural Storm Labs {__version__}</p>",
                        f"<div class='note'>{html.escape(DISCLAIMER)}</div>", _dossier_html(dossier), _triage_html(triage)]
    a = dossier.get("audio") or {}
    if a.get("available"):
        parts.append("<h2>Audio indicators</h2><ul>" + "".join(f"<li class='{f['level']}'><b>{f['level'].upper()}</b> {html.escape(f['text'])}</li>" for f in a["flags"]) + "</ul>")
        parts.append("<h2>Audio measurements</h2>" + _kv(a["metrics"]))
    elif dossier.get("audio_error"):
        parts.append(f"<h2>Audio</h2><p class='warn'>{html.escape(str(dossier['audio_error']))}</p>")
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Neural Storm Labs input report</title><style>{REPORT_CSS}</style></head><body>{''.join(parts)}</body></html>"
