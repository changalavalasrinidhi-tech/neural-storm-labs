"""Image forensics metrics: quality, noise, frequency spectrum, ELA, metadata, provenance markers.

All outputs are *indicators*, never proof. Every function degrades gracefully on odd inputs
(tiny images, grayscale, non-JPEG formats).
"""

from __future__ import annotations

import hashlib
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image

MAX_ANALYSIS_SIDE = 2048

# Keywords that, when found in metadata, point at editing or generation software.
EDITOR_KEYWORDS = ("photoshop", "gimp", "lightroom", "affinity", "pixelmator", "snapseed", "canva")
GENERATOR_KEYWORDS = (
    "stable diffusion", "midjourney", "dall-e", "dalle", "firefly", "comfyui", "automatic1111",
    "invokeai", "novelai", "leonardo", "imagen", "flux", "sdxl", "openai", "gemini",
)
# PNG text-chunk keys commonly written by local generation UIs.
GENERATOR_PNG_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata")


# ----------------------------------------------------------------------------- helpers
def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays / tuples into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj


def _prepare(rgb: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Return a uint8 RGB array (downscaled to MAX_ANALYSIS_SIDE) and the original (w, h) if scaled."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scaled_from = None
    if max(h, w) > MAX_ANALYSIS_SIDE:
        s = MAX_ANALYSIS_SIDE / float(max(h, w))
        arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        scaled_from = (w, h)
    return np.ascontiguousarray(arr), scaled_from


def _gray_f32(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


# ----------------------------------------------------------------------------- fingerprints
def file_fingerprint(raw: Optional[bytes]) -> Dict[str, Any]:
    if not raw:
        return {"size_bytes": None, "sha256": None, "md5": None}
    return {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "md5": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
    }


def perceptual_hashes(gray_u8: np.ndarray) -> Dict[str, str]:
    """64-bit average-hash and difference-hash as 16-char hex strings (robust to resizing)."""
    a = cv2.resize(gray_u8, (8, 8), interpolation=cv2.INTER_AREA)
    ahash = (a > a.mean()).flatten()
    d = cv2.resize(gray_u8, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    dhash = (d[:, 1:] > d[:, :-1]).flatten()

    def _hex(bits: np.ndarray) -> str:
        return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"

    return {"ahash": _hex(ahash), "dhash": _hex(dhash)}


def hamming_distance_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ----------------------------------------------------------------------------- pixel stats
def exposure_metrics(rgb: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    f = rgb.astype(np.float32)
    rg = f[..., 0] - f[..., 1]
    yb = 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    colorfulness = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    p1, p99 = np.percentile(gray, [1, 99])
    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "dynamic_range_p1_p99": float(p99 - p1),
        "clipped_shadows_pct": float((gray <= 2).mean() * 100),
        "clipped_highlights_pct": float((gray >= 253).mean() * 100),
        "saturation_mean": float(hsv[..., 1].mean() / 255.0),
        "colorfulness": float(colorfulness),
        "channel_means_rgb": [float(f[..., i].mean()) for i in range(3)],
    }


def dominant_colors(rgb: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(np.unique(small, axis=0))))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(small, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float64)
    order = np.argsort(-counts)
    return [
        {"hex": "#%02x%02x%02x" % tuple(int(c) for c in np.clip(centers[i], 0, 255)), "share_pct": float(counts[i] / counts.sum() * 100)}
        for i in order
    ]


def sharpness_metrics(gray: np.ndarray) -> Dict[str, Any]:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    if lap_var < 40:
        label = "very soft / blurred"
    elif lap_var < 120:
        label = "soft"
    elif lap_var < 800:
        label = "sharp"
    else:
        label = "very sharp / high detail"
    return {"laplacian_variance": lap_var, "tenengrad": tenengrad, "sharpness_label": label}


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise-variance estimator (std-dev in 0-255 gray levels)."""
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return float("nan")
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    sigma = math.sqrt(math.pi / 2.0) * float(np.abs(conv[1:-1, 1:-1]).sum()) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def entropy_bits(gray_u8: np.ndarray) -> float:
    hist = np.bincount(gray_u8.flatten(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def blockiness_ratio(gray: np.ndarray) -> float:
    """Mean |difference| across 8-px block boundaries over inside-block differences (JPEG grid cue)."""
    h, w = gray.shape[:2]
    if h < 24 or w < 24:
        return float("nan")
    dx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    cols = np.arange(dx.shape[1])
    boundary = dx[:, cols % 8 == 7].mean()
    inside = dx[:, cols % 8 != 7].mean() + 1e-6
    return float(boundary / inside)


# ----------------------------------------------------------------------------- frequency
def spectrum_metrics(gray: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """Radial power-spectrum statistics and a log-magnitude visualisation (uint8 RGB)."""
    h, w = gray.shape[:2]
    side = int(min(h, w, 512))
    side -= side % 2
    if side < 32:
        blank = np.zeros((64, 64, 3), np.uint8)
        return {"available": False, "reason": "image too small for spectrum analysis"}, blank
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = gray[y0:y0 + side, x0:x0 + side].astype(np.float32)
    crop = crop - crop.mean()
    win = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft2(crop * win))
    power = np.abs(spec) ** 2
    logmag = np.log1p(np.abs(spec))

    yy, xx = np.indices(power.shape)
    cy = cx = side // 2
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    rmax = side // 2
    radial_sum = np.bincount(r.flatten(), weights=power.flatten(), minlength=rmax + 1)[: rmax + 1]
    radial_cnt = np.maximum(np.bincount(r.flatten(), minlength=rmax + 1)[: rmax + 1], 1)
    radial = radial_sum / radial_cnt

    total = float(radial_sum[1:].sum() + 1e-12)
    hf_ratio = float(radial_sum[int(rmax * 0.5):].sum() / total)
    mid_ratio = float(radial_sum[int(rmax * 0.2): int(rmax * 0.5)].sum() / total)

    lo, hi = 4, max(int(rmax * 0.9), 8)
    rs = np.arange(lo, hi)
    vals = np.maximum(radial[lo:hi], 1e-12)
    slope = float(np.polyfit(np.log(rs), np.log(vals), 1)[0]) if len(rs) > 2 else float("nan")

    # Periodic peaks (grid / upsampling fingerprints) in the high-frequency annulus.
    resid = logmag - cv2.GaussianBlur(logmag, (0, 0), 3)
    annulus = (r > rmax * 0.25) & (r < rmax * 0.95)
    vals_a = resid[annulus]
    z = (resid - vals_a.mean()) / (vals_a.std() + 1e-9)
    peaks = (z > 6.0) & annulus
    n_peaks = int(cv2.connectedComponents(peaks.astype(np.uint8))[0] - 1)
    max_z = float(z[annulus].max())

    vis = (logmag - logmag.min()) / (np.ptp(logmag) + 1e-9)
    vis = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    vis[peaks] = (0, 255, 200)
    metrics = {
        "available": True,
        "analysed_crop_px": side,
        "high_freq_energy_ratio": hf_ratio,
        "mid_freq_energy_ratio": mid_ratio,
        "spectral_slope": slope,
        "periodic_peak_count": n_peaks,
        "max_peak_zscore": max_z,
        "radial_profile_db": [float(10 * np.log10(v + 1e-12)) for v in np.interp(np.linspace(1, rmax - 1, 64), np.arange(len(radial)), radial)],
    }
    return metrics, vis


# ----------------------------------------------------------------------------- ELA / noise map
def _tile_stats(m: np.ndarray, grid: int = 6) -> Tuple[float, float]:
    h, w = m.shape[:2]
    th, tw = max(h // grid, 1), max(w // grid, 1)
    means = [m[i * th:(i + 1) * th, j * tw:(j + 1) * tw].mean() for i in range(grid) for j in range(grid)
             if m[i * th:(i + 1) * th, j * tw:(j + 1) * tw].size]
    means_a = np.asarray(means, dtype=np.float64)
    cv = float(means_a.std() / (means_a.mean() + 1e-9))
    hot = float(means_a.max() / (np.median(means_a) + 1e-9))
    return cv, hot


def ela_analysis(rgb: np.ndarray, quality: int = 90) -> Tuple[Dict[str, Any], np.ndarray]:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    re = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    diff = np.abs(rgb.astype(np.int16) - re).mean(axis=2).astype(np.float32)
    p995 = float(np.percentile(diff, 99.5))
    scale = 255.0 / max(p995, 1.0)
    vis = cv2.applyColorMap(np.clip(diff * scale, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    cv, hot = _tile_stats(diff)
    return {
        "quality_used": int(quality),
        "mean_error": float(diff.mean()),
        "std_error": float(diff.std()),
        "p99_error": float(np.percentile(diff, 99)),
        "max_error": float(diff.max()),
        "tile_variation_cv": cv,
        "hottest_tile_ratio": hot,
    }, vis


def noise_residual_analysis(gray_u8: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    resid = gray_u8.astype(np.float32) - cv2.medianBlur(gray_u8, 3).astype(np.float32)
    mag = np.abs(resid)
    cv, hot = _tile_stats(mag)
    scale = 255.0 / max(float(np.percentile(mag, 99.5)), 1.0)
    vis = cv2.applyColorMap(np.clip(mag * scale, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return {
        "residual_std": float(resid.std()),
        "tile_inconsistency_cv": cv,
        "hottest_tile_ratio": hot,
    }, vis


# ----------------------------------------------------------------------------- metadata
_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def estimate_jpeg_quality(pil_img: Image.Image) -> Optional[int]:
    tables = getattr(pil_img, "quantization", None)
    if not tables:
        return None
    t0 = np.asarray(tables.get(0) or next(iter(tables.values())), dtype=np.float64)
    if t0.size != 64:
        return None
    best_q, best_err = None, 1e18
    for q in range(1, 101):
        s = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
        cand = np.clip(np.floor((_STD_LUMA_Q * s + 50.0) / 100.0), 1, 255)
        err = float(np.abs(cand - t0).sum())
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def metadata_report(raw: Optional[bytes]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "format": None, "exif": {}, "has_exif": False, "has_gps": False,
        "jpeg_quality_estimate": None, "png_text_keys": [], "generator_hits": [],
        "editor_hits": [], "c2pa_marker": False, "icc_profile": False,
    }
    if not raw:
        return out
    try:
        im = Image.open(io.BytesIO(raw))
    except Exception as exc:  # corrupt / unsupported
        out["error"] = f"could not parse container: {exc}"
        return out
    out["format"] = im.format
    out["icc_profile"] = bool(im.info.get("icc_profile"))
    out["jpeg_quality_estimate"] = estimate_jpeg_quality(im) if im.format == "JPEG" else None

    try:
        exif = im.getexif()
    except Exception:
        exif = None
    text_blob: List[str] = []
    if exif:
        merged = dict(exif)
        try:
            merged.update(dict(exif.get_ifd(0x8769)))
        except Exception:
            pass
        out["has_gps"] = bool(exif.get_ifd(0x8825)) if hasattr(exif, "get_ifd") else False
        for tag_id, value in merged.items():
            name = ExifTags.TAGS.get(tag_id, f"tag_{tag_id}")
            if isinstance(value, bytes):
                continue
            sval = str(value)
            if len(sval) > 120:
                sval = sval[:117] + "..."
            out["exif"][name] = sval
            text_blob.append(sval.lower())
        out["has_exif"] = bool(out["exif"])

    for key, value in (im.info or {}).items():
        if isinstance(key, str) and isinstance(value, (str, bytes)):
            if key.lower() in GENERATOR_PNG_KEYS:
                out["png_text_keys"].append(key)
            text_blob.append(str(value).lower()[:4000])
            text_blob.append(key.lower())

    blob = " ".join(text_blob)
    out["generator_hits"] = sorted({k for k in GENERATOR_KEYWORDS if k in blob} | set(out["png_text_keys"]))
    out["editor_hits"] = sorted({k for k in EDITOR_KEYWORDS if k in blob})
    # C2PA / Content Credentials marker: presence only, NOT cryptographic validation.
    head = raw[: min(len(raw), 2_000_000)]
    out["c2pa_marker"] = (b"c2pa" in head) or (b"jumb" in head and b"c2pa" in raw)
    return out


# ----------------------------------------------------------------------------- orchestrator
def forensic_flags(m: Dict[str, Any]) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []

    def add(level: str, text: str) -> None:
        flags.append({"level": level, "text": text})

    meta = m["metadata"]
    if meta.get("generator_hits"):
        add("warn", "Generation/editing-software markers found in metadata: " + ", ".join(meta["generator_hits"]) + ".")
    if meta.get("editor_hits"):
        add("info", "Editing software mentioned in metadata: " + ", ".join(meta["editor_hits"]) + ".")
    if meta.get("c2pa_marker"):
        add("info", "A C2PA / Content Credentials marker is present (presence only - signature NOT validated here).")
    if not meta.get("has_exif") and meta.get("format") in {"JPEG", "TIFF", None}:
        add("info", "No camera EXIF metadata (common after social-media re-encoding, but also typical of generated images).")
    if meta.get("has_gps"):
        add("info", "GPS location tags are embedded in this file (coordinates are not displayed or exported).")
    b = m["basic"]
    if min(b["width"], b["height"]) < 256:
        add("info", "Small image (<256 px short side): every forensic metric below is less reliable.")
    if b["width"] == b["height"] and b["width"] in {512, 640, 768, 1024, 1536, 2048}:
        add("weak", "Square power-of-two dimensions are typical of generative models (weak cue; many real crops share them).")
    megapixels = b["megapixels"]
    if megapixels >= 0.3 and m["noise"]["sigma"] < 1.0:
        add("warn", "Very low sensor-noise estimate: unusually smooth image (synthesis, heavy denoise or beauty filter).")
    sp = m["spectrum"]
    if sp.get("available") and sp["periodic_peak_count"] >= 4:
        add("warn", f"{sp['periodic_peak_count']} isolated high-frequency spectral peaks (possible resampling / upsampling grid; JPEG grids can also cause this).")
    ela = m["ela"]
    if meta.get("format") == "JPEG" and ela["tile_variation_cv"] > 0.6:
        add("info", "ELA error level varies strongly between regions - inspect the map; this is NOT proof of editing.")
    if meta.get("format") in {"PNG", "BMP", "TIFF", "WEBP"}:
        add("info", f"{meta['format']} input: ELA is only indicative for images that were once JPEG-compressed.")
    if not flags:
        add("info", "No notable metadata or low-level indicators were raised.")
    return flags


def analyze_image(rgb: np.ndarray, raw: Optional[bytes] = None, name: str = "image") -> Dict[str, Any]:
    """Run the full battery. Returns ``{"metrics": {...}, "maps": {...}, "flags": [...]}``."""
    img, scaled_from = _prepare(rgb)
    h, w = img.shape[:2]
    gray_f = _gray_f32(img)
    gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)

    metadata = metadata_report(raw)
    spectrum, spec_vis = spectrum_metrics(gray_f)
    ela, ela_vis = ela_analysis(img)
    resid, resid_vis = noise_residual_analysis(gray_u8)
    ow, oh = scaled_from if scaled_from else (w, h)
    metrics: Dict[str, Any] = {
        "name": name,
        "basic": {
            "width": ow, "height": oh, "megapixels": ow * oh / 1e6, "aspect_ratio": ow / max(oh, 1),
            "analysis_downscaled_from": list(scaled_from) if scaled_from else None,
            **file_fingerprint(raw),
        },
        "hashes": perceptual_hashes(gray_u8),
        "exposure": exposure_metrics(img),
        "dominant_colors": dominant_colors(img),
        "sharpness": sharpness_metrics(gray_f),
        "noise": {"sigma": noise_sigma(gray_f), "entropy_bits": entropy_bits(gray_u8), "blockiness_ratio": blockiness_ratio(gray_f)},
        "spectrum": spectrum,
        "ela": ela,
        "noise_residual": resid,
        "metadata": metadata,
    }
    metrics["flags"] = forensic_flags(metrics)
    return {"metrics": metrics, "maps": {"ela": ela_vis, "spectrum": spec_vis, "noise_residual": resid_vis}, "flags": metrics["flags"]}


# ----------------------------------------------------------------------------- face-vs-scene
def face_region_consistency(rgb: np.ndarray, box: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Compare sharpness / noise / brightness inside a face box with the ring around it.

    Face-swap composites often leave the face sharper or smoother than its surroundings.
    A mismatch is a weak cue on its own (depth of field and make-up also cause it).
    """
    img, _ = _prepare(rgb)
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 24 or bh < 24:
        return {"available": False, "reason": "face region too small"}
    px, py = int(bw * 0.5), int(bh * 0.5)
    ox1, oy1, ox2, oy2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
    gray = _gray_f32(img)
    mask = np.zeros(gray.shape, bool)
    mask[oy1:oy2, ox1:ox2] = True
    mask[y1:y2, x1:x2] = False
    if mask.sum() < 200:
        return {"available": False, "reason": "no surrounding area to compare"}
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    face_lap, ring_lap = float(lap[y1:y2, x1:x2].var()), float(lap[mask].var())
    face_noise = noise_sigma(gray[y1:y2, x1:x2])
    ring_crop = gray[oy1:oy2, ox1:ox2]
    ring_noise = noise_sigma(ring_crop)
    ratio = face_lap / (ring_lap + 1e-6)
    result = {
        "available": True,
        "face_laplacian_var": face_lap, "ring_laplacian_var": ring_lap, "sharpness_ratio_face_over_ring": float(ratio),
        "face_noise_sigma": face_noise, "ring_noise_sigma": ring_noise,
        "brightness_diff": float(gray[y1:y2, x1:x2].mean() - gray[mask].mean()),
        "face_area_pct": float(bw * bh / (W * H) * 100),
    }
    if ratio < 0.4 or ratio > 2.5:
        result["note"] = "Face and surroundings differ strongly in sharpness (weak face-swap / retouch cue)."
    else:
        result["note"] = "Face and surroundings have comparable sharpness."
    return result


# ----------------------------------------------------------------------------- video
def temporal_series(frames, max_frames: int = 48) -> Dict[str, Any]:
    """Per-frame quality series and frame-to-frame change for a list of RGB frames.

    Real footage drifts smoothly; spliced or regenerated frames can flicker in brightness,
    sharpness or noise. Flicker scores are weak cues on their own (cuts and auto-exposure also cause them).
    """
    frames = list(frames)
    if not frames:
        return {"available": False, "reason": "no frames"}
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    sharp, noise, bright, diffs = [], [], [], []
    prev = None
    for fr in frames:
        img, _ = _prepare(fr)
        g = _gray_f32(img)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        noise.append(noise_sigma(g))
        bright.append(float(g.mean()))
        small = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        diffs.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small

    def _flicker(series):
        a = np.asarray(series, dtype=np.float64)
        if len(a) < 3:
            return 0.0
        return float(np.std(np.diff(a)) / (np.abs(a).mean() + 1e-9))

    out = {
        "available": True, "n_frames": len(frames),
        "sharpness": sharp, "noise_sigma": noise, "brightness": bright, "frame_diff": diffs,
        "sharpness_flicker": _flicker(sharp), "noise_flicker": _flicker(noise), "brightness_flicker": _flicker(bright),
        "max_frame_diff": float(max(diffs)), "mean_frame_diff": float(np.mean(diffs[1:])) if len(diffs) > 1 else 0.0,
    }
    flags = []
    if out["sharpness_flicker"] > 0.35:
        flags.append("Sharpness fluctuates strongly between frames.")
    if out["brightness_flicker"] > 0.05:
        flags.append("Brightness flickers between frames.")
    if len(diffs) > 3 and out["max_frame_diff"] > 4 * (out["mean_frame_diff"] + 1e-6):
        flags.append("One frame differs far more from its neighbour than the rest (possible cut or splice).")
    out["flags"] = flags
    return out
