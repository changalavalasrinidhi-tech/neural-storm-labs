"""Input dossier + audio lab (torch-free).

Answers, for ANY submitted evidence, the first question a reviewer asks: *what exactly was given to
the system?*  ``profile_upload`` inspects an image, video, audio file, camera snapshot or microphone
clip and returns a plain dictionary: file facts, modality-specific facts, which analyses apply (and
why not, when they do not), and - for audio, including the audio track of a video - a full set of
signal measurements, indicator flags and plot data.

Only numpy / scipy / OpenCV / Pillow are required.  ffmpeg/ffprobe are used when present (richer
codec info, any audio format); everything degrades gracefully without them and says so.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
KIND_ICON = {"image": "🖼", "video": "🎞", "audio": "🎙"}
MAX_AUDIO_SECONDS = 30.0


# ----------------------------------------------------------------------------- small helpers
def human_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{n} B"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds:
        return "n/a"
    s = max(0.0, float(seconds))
    m, sec = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{sec:04.1f}" if h else f"{int(m):d}:{sec:04.1f}"


def detect_kind(name: str, mime: str = "") -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    for k in ("image", "audio", "video"):
        if (mime or "").startswith(k + "/"):
            return k
    return "video"


def _ffprobe(path: Path) -> Optional[Dict[str, Any]]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                             capture_output=True, timeout=20, check=False)
        return json.loads(out.stdout.decode("utf-8", "replace") or "{}") if out.returncode == 0 else None
    except Exception:
        return None


def _frac(txt: Any) -> Optional[float]:
    try:
        a, b = str(txt).split("/")
        return float(a) / float(b) if float(b) else None
    except Exception:
        try:
            return float(txt)
        except Exception:
            return None


def _write_temp(raw: bytes, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="ns_profile_", suffix=suffix or ".bin")
    with open(fd, "wb") as fh:  # closes the descriptor
        fh.write(raw)
    return Path(name)


# ----------------------------------------------------------------------------- audio decoding
def _decode_wav(path: Path, max_seconds: float) -> Optional[Tuple[np.ndarray, int, Dict[str, Any]]]:
    try:
        with wave.open(str(path), "rb") as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            frames = w.readframes(min(n, int(sr * max_seconds)))
        if sw == 1:
            x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            x = np.frombuffer(frames, "<i2").astype(np.float32) / 32768.0
        elif sw == 3:
            b = np.frombuffer(frames, np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float32) / 8388608.0
        elif sw == 4:
            x = np.frombuffer(frames, "<i4").astype(np.float32) / 2147483648.0
        else:
            return None
        x = x[: (len(x) // ch) * ch].reshape(-1, ch).mean(axis=1) if ch > 1 else x
        return x.astype(np.float32), int(sr), {"decoder": "python wave", "bit_depth": sw * 8, "channels": ch, "total_samples": int(n)}
    except Exception:
        return None


def _decode_ffmpeg(path: Path, sr_hint: Optional[int], max_seconds: float) -> Optional[Tuple[np.ndarray, int, Dict[str, Any]]]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-i", str(path), "-vn", "-t", f"{max_seconds:.1f}", "-ac", "1"]
    if sr_hint:
        cmd += ["-ar", str(int(sr_hint))]
    cmd += ["-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        x = np.frombuffer(out.stdout, "<f4").astype(np.float32)
        if out.returncode != 0 or x.size == 0:
            return None
        return x, int(sr_hint or 16000), {"decoder": "ffmpeg"}
    except Exception:
        return None


def _decode_soundfile(path: Path, max_seconds: float) -> Optional[Tuple[np.ndarray, int, Dict[str, Any]]]:
    try:
        import soundfile as sf  # type: ignore

        x, sr = sf.read(str(path), frames=-1, dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: int(sr * max_seconds)]
        return x.astype(np.float32), int(sr), {"decoder": "soundfile"}
    except Exception:
        return None


def decode_audio(path: Path, probe: Optional[Dict[str, Any]] = None,
                 max_seconds: float = MAX_AUDIO_SECONDS) -> Tuple[Optional[np.ndarray], int, Dict[str, Any]]:
    """Return (mono float32 samples, native sample rate, info). Samples are None when decoding is impossible."""
    sr_hint = None
    if probe:
        for s in probe.get("streams", []):
            if s.get("codec_type") == "audio":
                try:
                    sr_hint = int(s.get("sample_rate"))
                except Exception:
                    pass
                break
    if path.suffix.lower() == ".wav":
        r = _decode_wav(path, max_seconds)
        if r:
            return r
    r = _decode_ffmpeg(path, sr_hint, max_seconds) if sr_hint else None
    if r is None and sr_hint is None:
        r = _decode_ffmpeg(path, 16000, max_seconds)  # unknown rate: still analyse, at 16 kHz
        if r:
            r[2]["note"] = "native sample rate unknown (no ffprobe); decoded at 16 kHz"
    if r is None:
        r = _decode_soundfile(path, max_seconds)
    if r:
        return r
    return None, 0, {"error": "could not decode audio (install ffmpeg, or `pip install soundfile`)"}


# ----------------------------------------------------------------------------- audio analysis
def _frames(x: np.ndarray, n: int) -> np.ndarray:
    k = len(x) // n
    return x[: k * n].reshape(k, n) if k else np.zeros((0, n), np.float32)


def _db(v: Any, floor: float = 1e-10) -> Any:
    return 20.0 * np.log10(np.maximum(v, floor))


def _pitch_track(x: np.ndarray, sr: int, max_frames: int = 400) -> Dict[str, Any]:
    n = int(0.040 * sr)
    hop = int(0.020 * sr)
    if len(x) < n * 3 or sr < 8000:
        return {"voiced_ratio": 0.0, "f0_median_hz": None, "f0_std_hz": None}
    idx = np.arange(0, len(x) - n, hop)
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    win = np.hanning(n).astype(np.float32)
    fr = np.stack([x[i:i + n] * win for i in idx])
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    keep = _db(rms) > -45
    if keep.sum() < 3:
        return {"voiced_ratio": 0.0, "f0_median_hz": None, "f0_std_hz": None}
    fr = fr[keep]
    spec = np.fft.rfft(fr, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(sr / 400), int(sr / 60)
    seg = ac[:, lo:hi]
    peak, lag = seg.max(axis=1), seg.argmax(axis=1) + lo
    voiced = peak > 0.5
    f0 = sr / lag[voiced]
    f0 = f0[(f0 >= 60) & (f0 <= 400)]
    return {"voiced_ratio": float(voiced.mean()),
            "f0_median_hz": float(np.median(f0)) if f0.size >= 3 else None,
            "f0_std_hz": float(np.std(f0)) if f0.size >= 3 else None}


def analyze_audio(x: np.ndarray, sr: int, max_plot_cols: int = 360) -> Dict[str, Any]:
    """Measure a mono signal. Returns ``{"metrics", "flags", "viz"}`` (all JSON-friendly except ``viz`` arrays)."""
    from scipy import signal

    x = np.asarray(x, dtype=np.float32)
    if x.size < int(0.1 * sr) or sr <= 0:
        return {"available": False, "reason": "audio shorter than 0.1 s - nothing meaningful to measure"}
    dur = len(x) / sr
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    fl = int(0.020 * sr)
    fr = _frames(x, fl)
    frms = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-20)
    frms_db = _db(frms)
    active = frms_db > -50
    exact_zero = (np.abs(fr).max(axis=1) == 0)
    zero_runs, run, longest_zero = 0, 0, 0
    for z in exact_zero:
        run = run + 1 if z else 0
        longest_zero = max(longest_zero, run)
        if z and run == 3:  # >= 60 ms of digital silence
            zero_runs += 1
    jumps = 0
    if len(frms_db) > 2:
        d = np.abs(np.diff(frms_db))
        both = (frms_db[1:] > -60) & (frms_db[:-1] > -60)
        jumps = int(((d > 30) & both).sum())

    nper = 1024 if sr >= 16000 else 512
    f, t, S = signal.spectrogram(x, fs=sr, nperseg=nper, noverlap=nper // 2, mode="psd")
    S = S + 1e-14
    mean_psd = S.mean(axis=1)
    mean_db = 10 * np.log10(mean_psd)
    thr = mean_db.max() - 70.0
    above = np.where(mean_db > thr)[0]
    cutoff = float(f[above[-1]]) if above.size else float(f[-1])
    nyq = sr / 2.0
    cum = np.cumsum(mean_psd) / mean_psd.sum()
    centroid = float(((f[:, None] * S).sum(axis=0) / S.sum(axis=0)).mean())
    rolloff = float(f[min(np.searchsorted(cum, 0.85), len(f) - 1)])
    flat = float(np.mean(np.exp(np.mean(np.log(S), axis=0)) / S.mean(axis=0)))
    tot = mean_psd.sum()
    band = lambda lo, hi: float(mean_psd[(f >= lo) & (f < hi)].sum() / tot * 100)  # noqa: E731
    pitch = _pitch_track(x, sr)

    metrics: Dict[str, Any] = {
        "duration_s": dur, "sample_rate_hz": int(sr), "samples_analysed": int(len(x)),
        "peak_dbfs": float(_db(peak)), "rms_dbfs": float(_db(rms)), "crest_factor_db": float(_db(peak) - _db(rms)),
        "dc_offset": float(x.mean()), "clipping_pct": float((np.abs(x) >= 0.999).mean() * 100),
        "silence_pct": float((~active).mean() * 100) if len(active) else 0.0,
        "noise_floor_dbfs": float(np.percentile(frms_db, 10)) if len(frms_db) else None,
        "dynamic_range_db": float(np.percentile(frms_db, 95) - np.percentile(frms_db, 10)) if len(frms_db) else None,
        "zero_crossing_rate": float((np.diff(np.signbit(x)).sum()) / len(x)),
        "spectral_centroid_hz": centroid, "spectral_rolloff85_hz": rolloff, "spectral_flatness": flat,
        "bandwidth_cutoff_hz": cutoff, "nyquist_hz": nyq,
        "energy_below_300hz_pct": band(0, 300), "energy_300_3400hz_pct": band(300, 3400), "energy_above_8khz_pct": band(8000, nyq + 1),
        "voiced_ratio": pitch["voiced_ratio"], "f0_median_hz": pitch["f0_median_hz"], "f0_std_hz": pitch["f0_std_hz"],
        "digital_silence_runs": int(zero_runs), "longest_digital_silence_ms": float(longest_zero * 20),
        "abrupt_level_jumps": jumps,
    }

    flags: List[Dict[str, str]] = []

    def add(level: str, text: str) -> None:
        flags.append({"level": level, "text": text})

    if dur < 1.0:
        add("info", "Very short clip (<1 s): voice and spectral measurements are unreliable.")
    if metrics["clipping_pct"] > 1.0:
        add("warn", f"{metrics['clipping_pct']:.1f}% of samples are clipped (hard-limited) - distorted or heavily processed recording.")
    if metrics["silence_pct"] > 60:
        add("info", f"{metrics['silence_pct']:.0f}% of the clip is near-silent; little speech evidence is present.")
    if sr >= 22050 and cutoff < 0.75 * nyq and flat > 0.002:  # a pure tone is trivially 'band-limited'
        add("warn", f"Content stops at about {cutoff / 1000:.1f} kHz although the file is sampled at {sr / 1000:.1f} kHz - band-limited / up-sampled audio "
                    "(low-bitrate codec, phone line, or some speech synthesisers). Not proof on its own.")
    if zero_runs:
        add("warn", f"{zero_runs} run(s) of exact digital silence (>= 60 ms, longest {metrics['longest_digital_silence_ms']:.0f} ms). "
                    "Real rooms always have a noise floor, so this suggests editing, gating or synthesis.")
    if jumps >= 3:
        add("warn", f"{jumps} abrupt level jumps (>30 dB between adjacent 20 ms frames) - possible cuts/splices or aggressive gating.")
    if pitch["f0_std_hz"] is not None and pitch["voiced_ratio"] > 0.25 and pitch["f0_std_hz"] < 6:
        add("weak", f"Pitch is unusually steady (σ≈{pitch['f0_std_hz']:.1f} Hz): monotone delivery or a synthetic voice - sustained tones also do this.")
    if metrics["noise_floor_dbfs"] is not None and metrics["noise_floor_dbfs"] < -80 and dur >= 1:
        add("weak", f"Extremely clean noise floor ({metrics['noise_floor_dbfs']:.0f} dBFS): studio-grade, denoised or generated audio.")
    if sr in (16000, 22050, 24000):
        add("weak", f"Sample rate {sr} Hz is common for speech-synthesis output (also for phone/VoIP recordings).")
    if not flags:
        add("info", "No notable low-level audio indicators were raised.")

    # plot data (downsampled so the UI stays light)
    step = max(1, S.shape[1] // max_plot_cols)
    spec_db = 10 * np.log10(S[:, ::step])
    spec_db = np.clip(spec_db, spec_db.max() - 90, None)
    n_env = 1200
    seg = max(1, len(x) // n_env)
    env = np.abs(x[: (len(x) // seg) * seg]).reshape(-1, seg).max(axis=1)
    viz = {"envelope": env.astype(np.float32), "envelope_times": np.linspace(0, dur, len(env)),
           "spec_db": spec_db.astype(np.float32), "spec_freqs": f.astype(np.float32), "spec_times": t[::step].astype(np.float32),
           "psd_db": mean_db.astype(np.float32), "frame_rms_db": frms_db.astype(np.float32)}
    return {"available": True, "metrics": metrics, "flags": flags, "viz": viz}


# ----------------------------------------------------------------------------- modality facts
def _image_facts(raw: bytes) -> Dict[str, Any]:
    from PIL import Image

    out: Dict[str, Any] = {}
    try:
        im = Image.open(io.BytesIO(raw))
        out.update({"format": im.format, "colour_mode": im.mode, "width_px": im.size[0], "height_px": im.size[1],
                    "megapixels": round(im.size[0] * im.size[1] / 1e6, 2), "animated_frames": getattr(im, "n_frames", 1),
                    "has_alpha": im.mode in ("RGBA", "LA") or "transparency" in im.info})
        dpi = im.info.get("dpi")
        if dpi:
            out["dpi"] = ", ".join(f"{float(d):.0f}" for d in dpi)
    except Exception as exc:
        out["error"] = f"could not parse image: {exc}"
    return out


def _video_facts(path: Path, probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps, n = float(cap.get(cv2.CAP_PROP_FPS) or 0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        out.update({"width_px": w, "height_px": h, "fps": round(fps, 2), "frame_count": n,
                    "duration_s": round(n / fps, 2) if fps > 0 and n > 0 else None})
    except Exception as exc:
        out["error"] = f"OpenCV could not open the video: {exc}"
    if probe:
        fmt = probe.get("format", {})
        out["container"] = fmt.get("format_long_name") or fmt.get("format_name")
        try:
            out["duration_s"] = round(float(fmt["duration"]), 2)
            out["overall_bitrate_kbps"] = round(float(fmt["bit_rate"]) / 1000)
        except Exception:
            pass
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video" and "video_codec" not in out:
                out["video_codec"] = s.get("codec_name")
                out["pixel_format"] = s.get("pix_fmt")
                if _frac(s.get("avg_frame_rate")):
                    out["fps"] = round(_frac(s["avg_frame_rate"]) or 0, 2)
        out["has_audio_stream"] = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    return out


def _audio_stream_facts(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not probe:
        return {}
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            out = {"codec": s.get("codec_name"), "sample_rate_hz": s.get("sample_rate"), "channels": s.get("channels"),
                   "channel_layout": s.get("channel_layout")}
            if s.get("bit_rate"):
                try:
                    out["bitrate_kbps"] = round(float(s["bit_rate"]) / 1000)
                except Exception:
                    pass
            return {k: v for k, v in out.items() if v is not None}
    return {}


def _plan(kind: str, has_audio: Optional[bool], deep_available: Optional[bool]) -> List[Dict[str, Any]]:
    deep = "yes" if deep_available else "needs PyTorch (pip install -r requirements.txt)" if deep_available is False else "yes"
    rows: List[Dict[str, Any]] = []

    def add(stream: str, applies: bool, why: str) -> None:
        rows.append({"stream": stream, "applies": applies, "why": why})

    if kind == "image":
        add("Image forensics (ELA · FFT · noise · metadata)", True, "image pixels and file metadata are present")
        add("Face & hand landmarks", True, "runs on the still image")
        add("Deepfake model: visual stream", deep_available is not False, deep)
        add("Audio forensics", False, "an image carries no audio")
        add("Lip-sync (SyncNet)", False, "needs video: a moving face plus speech")
    elif kind == "audio":
        add("Audio forensics lab (signal measurements)", True, "audio samples are present")
        add("Voice-spoof model (WavLM / AASIST)", deep_available is not False, deep)
        add("Image / hand forensics", False, "no picture in an audio-only file")
        add("Lip-sync (SyncNet)", False, "needs video: a moving face plus speech")
    else:
        add("Frame lab + temporal consistency", True, "frames sampled evenly across the clip")
        add("Face & hand landmarks (per frame)", True, "runs on any frame you pick")
        add("Deepfake model: visual stream", deep_available is not False, deep)
        if has_audio is False:
            add("Audio forensics", False, "this video has no audio track")
            add("Voice-spoof model (WavLM / AASIST)", False, "this video has no audio track")
            add("Lip-sync (SyncNet)", False, "needs speech, and there is no audio track")
        else:
            add("Audio forensics lab", True, "audio track decoded from the video")
            add("Voice-spoof model (WavLM / AASIST)", deep_available is not False, deep)
            add("Lip-sync (SyncNet)", deep_available is not False, "runs if a face and speech are both found")
    return rows


# ----------------------------------------------------------------------------- public entry point
def profile_upload(name: str, raw: bytes, mime: str = "", source: str = "File upload",
                   deep_available: Optional[bool] = None) -> Dict[str, Any]:
    """Build the input dossier for one piece of evidence (see module docstring)."""
    kind = detect_kind(name, mime)
    ext = Path(name).suffix.lower()
    sha = hashlib.sha256(raw).hexdigest()
    prof: Dict[str, Any] = {
        "kind": kind, "icon": KIND_ICON[kind], "source": source, "name": name, "extension": ext or "(none)",
        "mime_declared": mime or "(browser sent none)", "size_bytes": len(raw), "size_human": human_size(len(raw)),
        "sha256": sha, "received_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "media": {}, "audio_stream": {}, "audio": None, "audio_error": None, "notes": [],
    }
    tmp: Optional[Path] = None
    try:
        if kind == "image":
            prof["media"] = _image_facts(raw)
        else:
            tmp = _write_temp(raw, ext)
            probe = _ffprobe(tmp)
            if probe is None and not shutil.which("ffprobe"):
                prof["notes"].append("ffprobe not found: codec details are limited (install ffmpeg for the full dossier).")
            if kind == "video":
                prof["media"] = _video_facts(tmp, probe)
                prof["audio_stream"] = _audio_stream_facts(probe)
            else:
                prof["audio_stream"] = _audio_stream_facts(probe)
                if probe:
                    try:
                        prof["media"]["duration_s"] = round(float(probe["format"]["duration"]), 2)
                        prof["media"]["container"] = probe["format"].get("format_long_name")
                    except Exception:
                        pass
            has_audio_stream = prof["media"].get("has_audio_stream")
            if kind == "audio" or has_audio_stream is not False:
                x, sr, info = decode_audio(tmp, probe)
                prof["audio_decode"] = info
                if x is not None:
                    res = analyze_audio(x, sr)
                    prof["audio"] = res
                    if kind == "audio" and not prof["media"].get("duration_s"):
                        prof["media"]["duration_s"] = round(len(x) / sr, 2)
                    if kind == "video" and has_audio_stream is None:
                        prof["media"]["has_audio_stream"] = True
                else:
                    prof["audio_error"] = info.get("error")
                    if kind == "video" and has_audio_stream is None:
                        prof["media"]["has_audio_stream"] = False
    except Exception as exc:  # never let profiling break the app
        prof["notes"].append(f"profiling error: {type(exc).__name__}: {exc}")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    has_audio = None
    if kind == "audio":
        has_audio = True
    elif kind == "video":
        has_audio = prof["media"].get("has_audio_stream")
    prof["plan"] = _plan(kind, has_audio, deep_available)
    prof["headline"] = headline(prof)
    return prof


def headline(p: Dict[str, Any]) -> str:
    m, parts = p["media"], [f"{p['icon']} {p['kind'].upper()}", p["name"], p["size_human"]]
    if p["kind"] == "image" and m.get("width_px"):
        parts.append(f"{m['width_px']}×{m['height_px']} px")
    if p["kind"] == "video":
        if m.get("width_px"):
            parts.append(f"{m['width_px']}×{m['height_px']}" + (f" @ {m['fps']:g} fps" if m.get("fps") else ""))
        parts.append(fmt_duration(m.get("duration_s")))
        parts.append("audio track: yes" if m.get("has_audio_stream") else "audio track: none")
    if p["kind"] == "audio":
        parts.append(fmt_duration(m.get("duration_s")))
        a = p.get("audio") or {}
        if a.get("available"):
            parts.append(f"{a['metrics']['sample_rate_hz'] / 1000:g} kHz")
    return " · ".join(str(x) for x in parts)


def profile_rows(p: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Table-ready rows: ``file``, ``media``, ``audio_stream``, ``plan``."""
    def kv(d: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"field": k.replace("_", " "), "value": "-" if v is None else (round(v, 3) if isinstance(v, float) else str(v))} for k, v in d.items()]

    file_rows = kv({"source": p["source"], "file name": p["name"], "extension": p["extension"], "declared MIME": p["mime_declared"],
                    "size": f"{p['size_human']} ({p['size_bytes']:,} bytes)", "SHA-256": p["sha256"], "received (UTC)": p["received_utc"]})
    return {"file": file_rows, "media": kv(p["media"]), "audio_stream": kv(p["audio_stream"]),
            "plan": [{"analysis": r["stream"], "runs": "✅ yes" if r["applies"] else "⛔ no", "why": r["why"]} for r in p["plan"]]}


def plain_summary(p: Dict[str, Any]) -> str:
    """One plain-English sentence stating exactly what was provided."""
    ext = p["extension"].lstrip(".").upper() or "unknown"
    k, m = p["kind"], p["media"]
    src = {"Camera snapshot": "a camera snapshot", "Microphone recording": "a microphone recording"}.get(p["source"])
    if k == "image":
        return f"You provided {src or 'a still image'} ({ext}) - pixels and file metadata only, no audio."
    if k == "audio":
        return f"You provided {src or 'an audio recording'} ({ext}) - sound only, no picture."
    track = "with an audio track" if m.get("has_audio_stream") else "without an audio track" if m.get("has_audio_stream") is False else "(audio track not determined)"
    return f"You provided a video ({ext}) {track} - moving pictures{' and sound' if m.get('has_audio_stream') else ''}."


def export_profile(p: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy of the dossier (plot arrays removed)."""
    from labs.image_metrics import to_jsonable

    q = {k: v for k, v in p.items() if k != "audio"}
    a = p.get("audio")
    if a:
        q["audio"] = {k: v for k, v in a.items() if k != "viz"}
    return to_jsonable(q)
