"""Built-in demo evidence so the app can be shown without any upload (torch-free)."""

from __future__ import annotations

import io
import wave

import numpy as np


def demo_voice_wav(seconds: float = 4.0, sr: int = 22050, seed: int = 7) -> bytes:
    """A voice-like synthetic clip with deliberately detectable traits: harmonic source with a steady
    pitch, hard band-limit at ~7 kHz, and a stretch of exact digital silence. Purely for demonstration."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 3 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 40) if k * 140 < 7000)
    syll = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * t) ** 2
    x = x * syll + 0.002 * rng.standard_normal(len(t))
    x[int(1.7 * sr):int(1.95 * sr)] = 0.0  # exact digital silence (an edit-like gap)
    x = x / (np.abs(x).max() + 1e-9) * 0.8
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class DemoUpload:
    """Duck-types a Streamlit UploadedFile."""

    def __init__(self, name: str, data: bytes, mime: str) -> None:
        self.name, self._d, self.type = name, data, mime

    def getvalue(self) -> bytes:
        return self._d

    def getbuffer(self) -> memoryview:
        return memoryview(self._d)

    def read(self) -> bytes:
        return self._d
