"""Synthetic 3-D hand skeletons for tests and the offline demo (NOT a detector, NOT real data)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Per-gesture finger bend (degrees at MCP, PIP, DIP). Extended = straight.
_STRAIGHT = (0.0, 0.0, 0.0)
_CURLED = (85.0, 100.0, 65.0)
_GESTURE_POSES: Dict[str, Dict[str, object]] = {
    "open": {"thumb": "out", "index": _STRAIGHT, "middle": _STRAIGHT, "ring": _STRAIGHT, "pinky": _STRAIGHT},
    "fist": {"thumb": "tucked", "index": _CURLED, "middle": _CURLED, "ring": _CURLED, "pinky": _CURLED},
    "point": {"thumb": "tucked", "index": _STRAIGHT, "middle": _CURLED, "ring": _CURLED, "pinky": _CURLED},
    "peace": {"thumb": "tucked", "index": _STRAIGHT, "middle": _STRAIGHT, "ring": _CURLED, "pinky": _CURLED},
    "three": {"thumb": "tucked", "index": _STRAIGHT, "middle": _STRAIGHT, "ring": _STRAIGHT, "pinky": _CURLED},
    "four": {"thumb": "tucked", "index": _STRAIGHT, "middle": _STRAIGHT, "ring": _STRAIGHT, "pinky": _STRAIGHT},
    "rock": {"thumb": "tucked", "index": _STRAIGHT, "middle": _CURLED, "ring": _CURLED, "pinky": _STRAIGHT},
    "thumbs_up": {"thumb": "up", "index": _CURLED, "middle": _CURLED, "ring": _CURLED, "pinky": _CURLED},
}
GESTURES = tuple(_GESTURE_POSES)

# (mcp_x_offset, spread_deg, [prox, mid, dist] lengths) in units of palm length.
_FINGER_GEOM = {
    "index": (-0.42, -6.0, (0.46, 0.27, 0.22)),
    "middle": (-0.14, -1.0, (0.50, 0.31, 0.23)),
    "ring": (0.14, 4.0, (0.46, 0.29, 0.22)),
    "pinky": (0.40, 10.0, (0.36, 0.21, 0.19)),
}


def make_synthetic_hand(gesture: str = "open", width: int = 640, height: int = 480,
                        center: Tuple[float, float] = (0.5, 0.5), palm_px: float = 160.0,
                        left_hand: bool = False) -> np.ndarray:
    """Return normalised ``(21, 3)`` landmarks for a hand facing the camera, fingers pointing up."""
    pose = _GESTURE_POSES[gesture]
    L = palm_px
    cx, cy = center[0] * width, center[1] * height
    wrist = np.array([cx, cy + 0.5 * L, 0.0])
    pts = np.zeros((21, 3))
    pts[0] = wrist

    def chain(start: np.ndarray, base_dir: np.ndarray, lengths: Sequence[float], bends: Tuple[float, float, float],
              curl_axis: np.ndarray) -> np.ndarray:
        out, pos, d = [], start.copy(), base_dir / np.linalg.norm(base_dir)
        phi = 0.0
        for k, ln in enumerate(lengths):
            phi += np.radians(bends[k])
            # rotate d about the in-plane x-axis toward the camera (negative z) by accumulated bend
            dd = d * np.cos(phi) + curl_axis * np.sin(phi)
            pos = pos + dd * ln * L
            out.append(pos.copy())
        return np.array(out)

    toward_cam = np.array([0.0, 0.0, -1.0])
    for name, (dx, spread, segs) in _FINGER_GEOM.items():
        mcp = wrist + np.array([dx * L, -1.0 * L if name == "middle" else -0.92 * L, 0.0])
        if name == "index":
            mcp[1] += 0.04 * L
        if name == "pinky":
            mcp[1] += 0.08 * L
        a = np.radians(spread)
        base = np.array([np.sin(a), -np.cos(a), 0.0])
        bends = pose[name]
        idx = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}[name]
        pts[idx] = mcp
        pts[idx + 1:idx + 4] = chain(mcp, base, segs, bends, toward_cam)

    # Thumb (explicit joint offsets in palm-length units, relative to the thumb CMC joint).
    cmc = wrist + np.array([-0.30 * L, -0.10 * L, 0.0])
    offsets = {
        "out": [(-0.20, -0.12, -0.02), (-0.22, -0.13, -0.03), (-0.20, -0.10, -0.03)],
        "tucked": [(0.05, -0.22, -0.04), (0.20, -0.14, -0.05), (0.17, -0.05, -0.05)],
        "up": [(-0.12, -0.30, -0.02), (-0.06, -0.30, -0.03), (-0.04, -0.30, -0.03)],
    }[str(pose["thumb"])]
    pts[1] = cmc
    cur = cmc.copy()
    for k, off in enumerate(offsets):
        cur = cur + np.array(off) * L
        pts[2 + k] = cur

    if left_hand:
        pts[:, 0] = 2 * cx - pts[:, 0]
    return np.stack([pts[:, 0] / width, pts[:, 1] / height, pts[:, 2] / width], axis=1)
