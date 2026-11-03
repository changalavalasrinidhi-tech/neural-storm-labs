"""Hand metrics from 21 MediaPipe-style landmarks: joint angles, proportions, gesture, plausibility.

Landmarks are an ``(21, 3)`` array of normalised x, y (0-1 of image width / height) and z
(relative depth, same scale as x). Nothing here identifies a person.

IMPORTANT limitation: landmark models regress a *canonical* 21-point skeleton for whatever they
perceive as a hand, so they tend to "repair" malformed hands. The plausibility checks therefore
catch skeleton-level oddities and low detector confidence, not every extra/merged finger.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FINGERS: Dict[str, Tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),   # CMC, MCP, IP, TIP
    "index": (5, 6, 7, 8),   # MCP, PIP, DIP, TIP
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
EXTENDED_ANGLE_DEG = 150.0

STATUS_SCORE = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` in degrees (180 = perfectly straight)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def to_pixel_space(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    return np.stack([lm[:, 0] * width, lm[:, 1] * height, lm[:, 2] * width], axis=1)


def finger_state(P: np.ndarray, name: str, palm_width: float) -> Dict[str, Any]:
    """Joint angles, length and extension flag for one finger (P in pixel space)."""
    a, b, c, d = FINGERS[name]
    wrist = P[0]
    seg = [_dist(P[a], P[b]), _dist(P[b], P[c]), _dist(P[c], P[d])]
    if name == "thumb":
        ang_mcp = _angle(P[1], P[2], P[3])
        ang_ip = _angle(P[2], P[3], P[4])
        # Rotation-invariant rule: IP joint fairly straight AND the tip is farther from the pinky
        # knuckle than the IP joint is, by a margin (a tucked thumb ends up near the palm centre).
        d_tip, d_ip = _dist(P[4], P[17]), _dist(P[3], P[17])
        extended = ang_ip > 145.0 and d_tip > d_ip and d_tip > 0.95 * palm_width
        return {
            "mcp_angle": ang_mcp, "pip_angle": ang_ip, "dip_angle": float("nan"),
            "length_px": sum(seg), "segments_px": seg, "extended": bool(extended),
            "curl_pct": float(np.clip((180.0 - ang_ip) / 90.0 * 100.0, 0, 100)),
        }
    ang_mcp = _angle(wrist, P[a], P[b])
    ang_pip = _angle(P[a], P[b], P[c])
    ang_dip = _angle(P[b], P[c], P[d])
    extended = ang_pip > EXTENDED_ANGLE_DEG and _dist(P[d], wrist) > _dist(P[b], wrist)
    flex = ((180.0 - ang_pip) + (180.0 - ang_dip)) / 2.0
    return {
        "mcp_angle": ang_mcp, "pip_angle": ang_pip, "dip_angle": ang_dip,
        "length_px": sum(seg), "segments_px": seg, "extended": bool(extended),
        "curl_pct": float(np.clip(flex / 90.0 * 100.0, 0, 100)),
    }


def classify_gesture(extended: Dict[str, bool], P: np.ndarray, palm_width: float) -> str:
    t, i, m, r, p = (extended[f] for f in FINGER_ORDER)
    others = (i, m, r, p)
    n_ext = sum(extended.values())
    if t and not any(others):
        return "Thumbs up" if P[4][1] < P[0][1] - 0.5 * palm_width else "Thumb out"
    if _dist(P[4], P[8]) < 0.45 * palm_width and m and r and p:
        return "OK sign"
    if n_ext == 5:
        return "Open palm"
    if n_ext == 0:
        return "Fist"
    if i and not (m or r or p):
        return "Pointing"
    if i and m and not (r or p):
        return "Peace / victory"
    if i and p and not (m or r):
        return "Rock"
    if t and p and not (i or m or r):
        return "Call me / shaka"
    if i and m and r and not p:
        return "Three"
    if i and m and r and p and not t:
        return "Four"
    return "Custom pose"


def analyze_hand(landmarks: np.ndarray, width: int, height: int, label: str = "Unknown",
                 score: Optional[float] = None) -> Dict[str, Any]:
    """Full metric set for one detected hand."""
    P = to_pixel_space(landmarks, width, height)
    palm_width = _dist(P[5], P[17])
    palm_length = _dist(P[0], P[9])
    fingers = {n: finger_state(P, n, palm_width) for n in FINGER_ORDER}
    extended = {n: fingers[n]["extended"] for n in FINGER_ORDER}
    gesture = classify_gesture(extended, P, palm_width)

    mid_len = fingers["middle"]["length_px"] + 1e-9
    for n in FINGER_ORDER:
        fingers[n]["length_rel_middle"] = fingers[n]["length_px"] / mid_len
        fingers[n]["length_rel_palm"] = fingers[n]["length_px"] / (palm_length + 1e-9)

    tips = [P[FINGERS[n][3]] for n in FINGER_ORDER]
    span = max(_dist(a, b) for a, b in combinations(tips, 2))
    dirs = {n: P[FINGERS[n][3]] - P[FINGERS[n][0]] for n in ("index", "middle", "ring", "pinky")}

    def ang_between(u: np.ndarray, v: np.ndarray) -> float:
        n = np.linalg.norm(u) * np.linalg.norm(v)
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / n, -1, 1)))) if n > 1e-9 else 0.0

    spread = {
        "index_middle": ang_between(dirs["index"], dirs["middle"]),
        "middle_ring": ang_between(dirs["middle"], dirs["ring"]),
        "ring_pinky": ang_between(dirs["ring"], dirs["pinky"]),
        "thumb_index": ang_between(P[4] - P[2], P[8] - P[5]),
    }
    up = P[9][:2] - P[0][:2]
    orientation = float(np.degrees(math.atan2(up[0], -up[1])))  # 0 deg = fingers pointing up
    xy = P[:, :2]
    bbox = (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))
    openness = float(np.mean([_dist(P[FINGERS[n][3]], P[0]) for n in ("index", "middle", "ring", "pinky")]) / (palm_length + 1e-9))

    metrics: Dict[str, Any] = {
        "label": label,
        "detector_score": score,
        "gesture": gesture,
        "finger_count": int(sum(extended.values())),
        "extended": extended,
        "palm_width_px": palm_width,
        "palm_length_px": palm_length,
        "palm_aspect": palm_length / (palm_width + 1e-9),
        "hand_span_px": span,
        "openness_index": openness,
        "orientation_deg": orientation,
        "bbox_px": bbox,
        "bbox_area_pct": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height) * 100.0),
        "spread_deg": spread,
        "fingers": fingers,
        "mean_depth_z": float(np.mean(np.asarray(landmarks, dtype=np.float64).reshape(21, 3)[:, 2])),
    }
    metrics["plausibility"] = plausibility(metrics, np.asarray(landmarks, dtype=np.float64).reshape(21, 3))
    return metrics


def _band(value: float, lo: float, hi: float, slack: float) -> str:
    if lo <= value <= hi:
        return "pass"
    if lo - slack <= value <= hi + slack:
        return "warn"
    return "fail"


def plausibility(m: Dict[str, Any], lm_norm: np.ndarray) -> Dict[str, Any]:
    """Heuristic skeleton plausibility (0-100). A screening aid - see module docstring."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, value: Any, expected: str, status: str, weight: float = 1.0) -> None:
        checks.append({"check": name, "value": value, "expected": expected, "status": status, "weight": weight})

    sc = m.get("detector_score")
    if sc is not None:
        add("Detector confidence", round(float(sc), 3), ">= 0.80", "pass" if sc >= 0.8 else "warn" if sc >= 0.6 else "fail", 1.5)

    inside = float(np.mean((lm_norm[:, 0] > -0.02) & (lm_norm[:, 0] < 1.02) & (lm_norm[:, 1] > -0.02) & (lm_norm[:, 1] < 1.02)))
    add("Landmarks inside frame", f"{inside * 100:.0f}%", "100% (cropped hands are less reliable)", "pass" if inside >= 0.999 else "warn" if inside >= 0.8 else "fail")
    area = m["bbox_area_pct"]
    add("Hand size in frame", f"{area:.2f}% of image", ">= 1% for stable landmarks", "pass" if area >= 1.0 else "warn" if area >= 0.3 else "fail")

    ranges = {"index": (0.75, 1.10, 0.10), "ring": (0.80, 1.10, 0.10), "pinky": (0.55, 0.95, 0.10), "thumb": (0.50, 1.15, 0.12)}
    for n, (lo, hi, slack) in ranges.items():
        v = m["fingers"][n]["length_rel_middle"]
        add(f"{n.capitalize()}/middle length", round(v, 3), f"{lo:.2f}-{hi:.2f}", _band(v, lo, hi, slack))

    pa = m["palm_aspect"]
    add("Palm length/width", round(pa, 3), "0.80-1.70", _band(pa, 0.8, 1.7, 0.2))

    for k in ("index_middle", "middle_ring", "ring_pinky"):
        v = m["spread_deg"][k]
        add(f"Finger spread {k.replace('_', '-')}", f"{v:.1f} deg", "<= 45 deg", "pass" if v <= 45 else "warn" if v <= 60 else "fail")
    ti = m["spread_deg"]["thumb_index"]
    add("Thumb-index angle", f"{ti:.1f} deg", "<= 110 deg", "pass" if ti <= 110 else "warn" if ti <= 130 else "fail")

    bad_seg = []
    for n in ("index", "middle", "ring", "pinky"):
        s = m["fingers"][n]["segments_px"]
        if s[0] > 1e-6 and not (0.4 <= s[1] / s[0] <= 1.1 and 0.4 <= s[2] / max(s[1], 1e-6) <= 1.3):
            bad_seg.append(n)
    add("Phalanx proportions", "ok" if not bad_seg else "odd: " + ", ".join(bad_seg), "proximal >= middle >= ~distal", "pass" if not bad_seg else "warn" if len(bad_seg) == 1 else "fail")

    total_w = sum(c["weight"] for c in checks)
    score = 100.0 * sum(STATUS_SCORE[c["status"]] * c["weight"] for c in checks) / total_w
    return {
        "score": float(score),
        "label": "plausible skeleton" if score >= 80 else "some oddities" if score >= 55 else "implausible skeleton",
        "checks": checks,
        "n_warn": sum(c["status"] == "warn" for c in checks),
        "n_fail": sum(c["status"] == "fail" for c in checks),
    }


def scene_checks(hands: Sequence[Dict[str, Any]], n_faces: int) -> List[Dict[str, str]]:
    """Cross-hand and hand-vs-face observations for the whole image."""
    out: List[Dict[str, str]] = []
    if not hands:
        return [{"level": "info", "text": "No hands detected."}]
    if n_faces == 1 and len(hands) > 2:
        out.append({"level": "warn", "text": f"{len(hands)} hands for 1 face: another person, a reflection, or a duplicated-hand artifact."})
    labels = [h["label"] for h in hands if h["label"] in {"Left", "Right"}]
    if len(labels) >= 2 and len(set(labels)) == 1 and len(hands) == 2:
        out.append({"level": "info", "text": f"Both hands are labelled '{labels[0]}' - possible with two people or a mirrored image; also a known generator slip."})
    low = [h for h in hands if h.get("detector_score") is not None and h["detector_score"] < 0.6]
    if low:
        out.append({"level": "warn", "text": f"{len(low)} hand(s) detected with low confidence (<0.60)."})
    if not out:
        out.append({"level": "info", "text": "No cross-hand inconsistencies noticed."})
    return out
