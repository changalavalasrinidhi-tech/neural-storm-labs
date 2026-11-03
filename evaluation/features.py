from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from labs.image_metrics import analyze_image

NAMES = []
def vector(rgb):
    r = analyze_image(rgb)
    m = r["metrics"]
    d = {
        "lap_var": np.log1p(m["sharpness"]["laplacian_variance"]), "tenengrad": np.log1p(m["sharpness"]["tenengrad"]),
        "sigma": m["noise"]["sigma"], "entropy": m["noise"]["entropy_bits"], "block": m["noise"]["blockiness_ratio"],
        "hf": np.log(m["spectrum"].get("high_freq_energy_ratio", 0) + 1e-9), "mf": np.log(m["spectrum"].get("mid_freq_energy_ratio", 0) + 1e-9),
        "slope": m["spectrum"].get("spectral_slope", 0), "peaks": m["spectrum"].get("periodic_peak_count", 0),
        "peakz": m["spectrum"].get("max_peak_zscore", 0),
        "ela_mean": m["ela"]["mean_error"], "ela_std": m["ela"]["std_error"], "ela_cv": m["ela"]["tile_variation_cv"],
        "ela_hot": m["ela"]["hottest_tile_ratio"],
        "res_std": m["noise_residual"]["residual_std"], "res_cv": m["noise_residual"]["tile_inconsistency_cv"],
        "res_hot": m["noise_residual"]["hottest_tile_ratio"],
        "sat": m["exposure"]["saturation_mean"], "colorful": m["exposure"]["colorfulness"],
    }
    warn = sum(f["level"] == "warn" for f in r["flags"])
    return d, warn
