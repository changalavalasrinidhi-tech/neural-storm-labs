"""Review-priority triage score (0-100) with a fully transparent breakdown (torch-free).

This is NOT a probability that something is fake. It answers: "how many measurable indicators did
the evidence raise, so how urgently should a human look at it?"  Every point is traceable to a row
in the breakdown table, and the score also reports what was *not* measured (coverage).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

BANDS = ((60, "High", "Several independent indicators - review this first."),
         (30, "Medium", "Some indicators - worth a closer human look."),
         (0, "Low", "Few measurable indicators - this does not prove the media is authentic."))

PER_WARN = 8
CAPS = {"image": 40, "hands": 15, "faces": 10, "audio": 30, "video": 18}


def _row(signal: str, points: float, why: str) -> Dict[str, Any]:
    return {"signal": signal, "points": round(float(points), 1), "why": why}


def triage_score(image_flags: Optional[Sequence[Dict[str, str]]] = None,
                 hands: Optional[Sequence[Dict[str, Any]]] = None,
                 faces: Optional[Sequence[Dict[str, Any]]] = None,
                 audio_flags: Optional[Sequence[Dict[str, str]]] = None,
                 temporal_flags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    coverage: List[str] = []

    if image_flags is not None:
        coverage.append("image forensics")
        n = sum(f["level"] == "warn" for f in image_flags)
        pts = min(CAPS["image"], n * PER_WARN)
        if pts:
            rows.append(_row("Image forensics", pts, f"{n} warning-level indicator(s) (metadata, noise, spectrum)"))
    if hands is not None:
        coverage.append("hands")
        low = [h for h in hands if h["plausibility"]["score"] < 70]
        if low:
            worst = min(h["plausibility"]["score"] for h in low)
            rows.append(_row("Hand plausibility", min(CAPS["hands"], (70 - worst) / 3 + 3), f"{len(low)} hand(s) below 70/100 (lowest {worst:.0f})"))
    if faces is not None:
        coverage.append("faces")
        odd = [f for f in faces if f.get("consistency", {}).get("available")
               and not str(f["consistency"].get("note", "")).startswith("Face and surroundings have comparable")]
        if odd:
            rows.append(_row("Face vs surroundings", min(CAPS["faces"], 5 * len(odd)), f"{len(odd)} face(s) differ strongly from their surroundings"))
    if audio_flags is not None:
        coverage.append("audio")
        n = sum(f["level"] == "warn" for f in audio_flags)
        w = sum(f["level"] == "weak" for f in audio_flags)
        pts = min(CAPS["audio"], n * 7 + w * 2)
        if pts:
            rows.append(_row("Audio forensics", pts, f"{n} warning + {w} weak audio indicator(s)"))
    if temporal_flags is not None:
        coverage.append("video temporal")
        pts = min(CAPS["video"], 6 * len(temporal_flags))
        if pts:
            rows.append(_row("Video temporal consistency", pts, f"{len(temporal_flags)} frame-to-frame indicator(s)"))

    score = float(min(100.0, sum(r["points"] for r in rows)))
    label, advice = next((b[1], b[2]) for b in BANDS if score >= b[0])
    return {"score": round(score, 1), "label": label, "advice": advice, "breakdown": rows, "coverage": coverage,
            "not_measured": [m for m in ("image forensics", "hands", "faces", "audio", "video temporal") if m not in coverage],
            "disclaimer": "A review-priority aid, not a probability of manipulation."}
