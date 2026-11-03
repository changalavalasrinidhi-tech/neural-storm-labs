from __future__ import annotations
import numpy as np, cv2
from scipy.stats import kurtosis
from features import vector

def _hp(g):
    return g - cv2.GaussianBlur(g, (0, 0), 1.0)

def extra(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = _hp(g); sd = r.std() + 1e-6
    d = {"res_kurt": float(kurtosis(r.ravel())), "res_sd_log": float(np.log(sd))}
    # residual autocorrelation at periodic lags (resampling / transposed-conv grids)
    for lag in (1, 2, 3, 4, 8):
        d[f"ac_x{lag}"] = float((r[:, :-lag] * r[:, lag:]).mean() / sd**2)
        d[f"ac_y{lag}"] = float((r[:-lag] * r[lag:]).mean() / sd**2)
    # fine-scale vs coarse-scale detail ratios (smoothing / upsampling kills fine detail)
    prev = g
    for s in (0.8, 1.5, 3.0):
        b = cv2.GaussianBlur(g, (0, 0), s); d[f"band_{s}"] = float(np.log((g - b).std() + 1e-6)); 
    d["fine_over_coarse"] = d["band_0.8"] - d["band_3.0"]
    # radial log-spectrum, 12 bins, normalised by low-frequency level
    F = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])))))
    h, w = F.shape; yy, xx = np.mgrid[:h, :w]; rad = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    bins = np.linspace(0.02, 1.0, 13); prof = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (rad >= a) & (rad < b); prof.append(np.log(F[m].mean() + 1e-6))
    prof = np.array(prof); prof -= prof[0]
    for i, v in enumerate(prof[1:]): d[f"rad{i}"] = float(v)
    d["rad_top_resid"] = float(prof[-1] - np.polyval(np.polyfit(np.arange(12), prof, 1), 11))
    # local-variance heterogeneity (splices / blended regions)
    lv = cv2.blur(r**2, (16, 16))[::16, ::16]
    d["lv_cv"] = float(lv.std() / (lv.mean() + 1e-6)); d["lv_max_ratio"] = float(lv.max() / (np.median(lv) + 1e-6))
    # cross-channel residual correlation (colour interpolation / generator colour coupling)
    rr = [_hp(rgb[..., c].astype(np.float32)).ravel() for c in range(3)]
    d["chan_corr"] = float(np.mean([np.corrcoef(rr[i], rr[j])[0, 1] for i, j in ((0, 1), (0, 2), (1, 2))]))
    return d

def full_vector(rgb):
    d, warn = vector(rgb); d.update(extra(rgb)); return d, warn
