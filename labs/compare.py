"""Side-by-side tamper comparison of two images (torch-free).

Typical use: an original and a suspected edit/re-post. Reports pixel change, structural similarity,
perceptual-hash distance and the regions that changed most, plus a heat-map overlay.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from labs import image_metrics as im


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    saa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())


def compare_images(a_rgb: np.ndarray, b_rgb: np.ndarray, max_side: int = 900, thresh: int = 25) -> Dict[str, Any]:
    h, w = a_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    A = cv2.resize(a_rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    resized = b_rgb.shape[:2] != a_rgb.shape[:2]
    B = cv2.resize(b_rgb, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(cv2.GaussianBlur(ga, (3, 3), 0), cv2.GaussianBlur(gb, (3, 3), 0))
    changed = diff > thresh
    ssim = _ssim(ga, gb)
    ha, hb = im.perceptual_hashes(ga), im.perceptual_hashes(gb)
    hamming = {k: im.hamming_distance_hex(ha[k], hb[k]) for k in ha if k in hb}

    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.cvtColor(cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(A, 0.55, heat, 0.45, 0)

    mask = cv2.morphologyEx(changed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict[str, Any]] = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 0.002 * A.shape[0] * A.shape[1]:
            continue
        regions.append({"x": x, "y": y, "w": bw, "h": bh, "area_pct": round(bw * bh / (A.shape[0] * A.shape[1]) * 100, 2),
                        "mean_diff": round(float(diff[y:y + bh, x:x + bw].mean()), 1)})
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (97, 242, 194), 2)

    changed_pct = float(changed.mean() * 100)
    if ssim > 0.995 and changed_pct < 0.1:
        verdict = "Near-identical: any difference is invisible re-encoding noise."
    elif regions and changed_pct < 25 and ssim > 0.75:
        verdict = "Localised differences: the highlighted regions changed while the rest matches - inspect them (edit, retouch or overlay)."
    elif ssim > 0.9:
        verdict = "Globally similar with scattered small changes - typical of re-compression, resizing or colour grading."
    else:
        verdict = "Substantially different images."
    return {"ssim": round(ssim, 4), "changed_pixels_pct": round(changed_pct, 2), "mean_abs_diff": round(float(diff.mean()), 2),
            "max_abs_diff": int(diff.max()), "hash_hamming": hamming, "regions": regions, "second_image_resized": bool(resized),
            "verdict": verdict, "heatmap": heat, "overlay": overlay, "diff_gray": norm}
