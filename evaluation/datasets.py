"""Offline evaluation data: REAL photos (bundled with scikit-image / scikit-learn) vs SYNTHETIC fakes.

IMPORTANT: the 'fake' class is produced by *our own* artifact simulators, not by real deepfake
generators. Accuracy measured here proves the metrics can see these artifact families; it is NOT an
estimate of accuracy on real-world deepfakes (FaceForensics++, DFDC, ...).
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from skimage import data as skd

REAL_SOURCES = ["astronaut", "camera", "coffee", "chelsea", "rocket", "moon", "retina",
                "hubble_deep_field", "grass", "gravel", "brick", "text"]

def _to_rgb(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.uint8)
    return a[..., :3]

def load_real_sources():
    out = {}
    for n in REAL_SOURCES:
        try:
            out[n] = _to_rgb(getattr(skd, n)())
        except Exception:
            pass
    import sklearn, glob
    d = os.path.join(os.path.dirname(sklearn.__file__), "datasets", "images")
    for f in glob.glob(os.path.join(d, "*.jpg")):
        out["sk_" + os.path.basename(f)[:-4]] = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    return out

def crops(img, n, size, rng):
    h, w = img.shape[:2]
    size = min(size, h, w)
    res = []
    for _ in range(n):
        y = rng.integers(0, h - size + 1); x = rng.integers(0, w - size + 1)
        res.append(img[y:y + size, x:x + size].copy())
    return res

# ------------------------------------------------------------------ fake simulators
def fake_upsample(a, rng):          # generator decoders: low-res synthesis then learned/nearest upsample
    h, w = a.shape[:2]; f = rng.choice([2, 3, 4])
    s = cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST if rng.random() < .5 else cv2.INTER_CUBIC)

def fake_grid(a, rng):              # transposed-conv checkerboard / periodic spectral peaks
    h, w = a.shape[:2]; p = int(rng.choice([2, 4, 8])); amp = rng.uniform(2.5, 6)
    yy, xx = np.mgrid[:h, :w]
    pat = np.cos(2 * np.pi * xx / p) * np.cos(2 * np.pi * yy / p) * amp
    return np.clip(a.astype(np.float32) + pat[..., None], 0, 255).astype(np.uint8)

def fake_smooth(a, rng):            # over-smooth / beauty-filter / diffusion-style denoised look
    k = int(rng.choice([5, 7, 9]))
    b = cv2.bilateralFilter(a, k, 60, 60)
    return cv2.GaussianBlur(b, (0, 0), rng.uniform(0.8, 1.6))

def fake_splice(a, rng, donors):    # face-swap-like paste of a region with different noise/blur statistics
    h, w = a.shape[:2]; out = a.copy()
    d = donors[rng.integers(len(donors))]
    d = cv2.resize(d, (w, h))
    d = cv2.GaussianBlur(d, (0, 0), rng.uniform(1.0, 2.2))
    ph, pw = int(h * rng.uniform(.3, .5)), int(w * rng.uniform(.3, .5))
    y, x = rng.integers(0, h - ph), rng.integers(0, w - pw)
    m = np.zeros((h, w), np.float32); m[y:y + ph, x:x + pw] = 1
    m = cv2.GaussianBlur(m, (0, 0), 4)[..., None]
    return (out * (1 - m) + d * m).astype(np.uint8)

FAKE_KINDS = ["upsample", "grid", "smooth", "splice"]

def make_fake(kind, a, rng, donors):
    return {"upsample": fake_upsample, "grid": fake_grid, "smooth": fake_smooth}.get(kind, None)(a, rng) \
        if kind != "splice" else fake_splice(a, rng, donors)
