"""Tests for input dossier, audio lab, triage, custody chain, compare and demo. Run: python -m unittest discover -s tests -v"""

import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs import compare, custody, demo, input_profile as ip, report, triage  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _png(rgb) -> bytes:
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "PNG"); return b.getvalue()


def _wav(x: np.ndarray, sr: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


def _ffmpeg(args: str, suffix: str) -> bytes:
    out = Path(tempfile.mkdtemp()) / ("o" + suffix)
    subprocess.run(f"ffmpeg -v error -y {args} {out}", shell=True, check=True)
    data = out.read_bytes(); shutil.rmtree(out.parent, ignore_errors=True); return data


class HelpersTests(unittest.TestCase):
    def test_kind_detection(self):
        self.assertEqual(ip.detect_kind("a.JPG"), "image")
        self.assertEqual(ip.detect_kind("a.mp3"), "audio")
        self.assertEqual(ip.detect_kind("a.mov"), "video")
        self.assertEqual(ip.detect_kind("noext", "audio/wav"), "audio")

    def test_formatting(self):
        self.assertEqual(ip.human_size(512), "512 B")
        self.assertEqual(ip.human_size(1536), "1.5 KB")
        self.assertEqual(ip.fmt_duration(65.2), "1:05.2")
        self.assertEqual(ip.fmt_duration(None), "n/a")


class ProfileTests(unittest.TestCase):
    def test_image_profile(self):
        rgb = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        p = ip.profile_upload("x.png", _png(rgb), "image/png")
        self.assertEqual(p["kind"], "image")
        self.assertEqual((p["media"]["width_px"], p["media"]["height_px"]), (160, 120))
        self.assertIn("160×120", p["headline"])
        plan = {r["stream"]: r["applies"] for r in p["plan"]}
        self.assertFalse(plan["Audio forensics"])
        self.assertIn("no audio", ip.plain_summary(p))
        self.assertEqual(len(p["sha256"]), 64)

    def test_camera_and_mic_wording(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        self.assertIn("camera snapshot", ip.plain_summary(ip.profile_upload("c.png", _png(rgb), source="Camera snapshot")))
        self.assertIn("microphone recording", ip.plain_summary(ip.profile_upload("m.wav", demo.demo_voice_wav(), source="Microphone recording")))

    def test_audio_profile_and_json(self):
        p = ip.profile_upload("v.wav", demo.demo_voice_wav(), "audio/wav", deep_available=False)
        self.assertEqual(p["kind"], "audio")
        self.assertTrue(p["audio"]["available"])
        self.assertAlmostEqual(p["media"]["duration_s"], 4.0, delta=0.1)
        plan = {r["stream"]: r for r in p["plan"]}
        self.assertTrue(plan["Audio forensics lab (signal measurements)"]["applies"])
        self.assertFalse(plan["Voice-spoof model (WavLM / AASIST)"]["applies"])      # deep_available False
        self.assertIn("PyTorch", plan["Voice-spoof model (WavLM / AASIST)"]["why"])
        exported = ip.export_profile(p)
        self.assertNotIn("viz", exported["audio"])
        json.dumps(exported)                                                          # must be JSON-safe

    def test_corrupt_audio_is_reported_not_raised(self):
        p = ip.profile_upload("bad.wav", b"RIFF....", "audio/wav")
        self.assertIsNone(p["audio"])
        self.assertTrue(p["audio_error"] or p["notes"])

    def test_too_short_audio(self):
        p = ip.profile_upload("s.wav", _wav(np.zeros(500), 16000), "audio/wav")
        self.assertFalse(p["audio"]["available"])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_video_with_and_without_audio(self):
        v = _ffmpeg("-f lavfi -i testsrc=duration=2:size=160x120:rate=10 -f lavfi -i sine=frequency=300:duration=2 -c:v mpeg4 -c:a aac", ".mp4")
        s = _ffmpeg("-f lavfi -i testsrc=duration=2:size=160x120:rate=10 -c:v mpeg4", ".mp4")
        pv, ps = ip.profile_upload("v.mp4", v, "video/mp4"), ip.profile_upload("s.mp4", s, "video/mp4")
        self.assertTrue(pv["media"]["has_audio_stream"]); self.assertTrue(pv["audio"]["available"])
        self.assertIn("audio track: yes", pv["headline"])
        self.assertFalse(ps["media"]["has_audio_stream"]); self.assertIsNone(ps["audio"])
        self.assertIn("audio track: none", ps["headline"])
        reasons = {r["stream"]: r["why"] for r in ps["plan"]}
        self.assertIn("no audio track", reasons["Audio forensics"])
        self.assertEqual((pv["media"]["width_px"], pv["media"]["height_px"]), (160, 120))

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_mp3_decodes(self):
        m = _ffmpeg("-f lavfi -i anoisesrc=d=2:c=white:r=44100 -b:a 96k", ".mp3")
        p = ip.profile_upload("n.mp3", m, "audio/mpeg")
        self.assertTrue(p["audio"]["available"])
        self.assertEqual(p["audio"]["metrics"]["sample_rate_hz"], 44100)


class AudioAnalysisTests(unittest.TestCase):
    def levels(self, x, sr):
        r = ip.analyze_audio(x, sr)
        return r, [f["level"] for f in r["flags"]]

    def test_demo_clip_raises_expected_indicators(self):
        p = ip.profile_upload("d.wav", demo.demo_voice_wav())
        texts = " ".join(f["text"] for f in p["audio"]["flags"])
        self.assertIn("digital silence", texts)
        self.assertIn("band-limited", texts)
        m = p["audio"]["metrics"]
        self.assertAlmostEqual(m["f0_median_hz"], 140, delta=6)
        self.assertGreaterEqual(m["digital_silence_runs"], 1)

    def test_clipping_detected(self):
        t = np.arange(16000 * 2) / 16000
        x = np.clip(3.0 * np.sin(2 * np.pi * 220 * t), -1, 1)
        r = ip.analyze_audio(x.astype(np.float32), 16000)
        self.assertGreater(r["metrics"]["clipping_pct"], 20)
        self.assertTrue(any("clipped" in f["text"] for f in r["flags"]))

    def test_natural_noise_is_not_flagged_as_edited(self):
        x = (0.1 * np.random.default_rng(3).standard_normal(44100 * 2)).astype(np.float32)
        r = ip.analyze_audio(x, 44100)
        self.assertEqual(r["metrics"]["digital_silence_runs"], 0)
        self.assertFalse(any(f["level"] == "warn" for f in r["flags"]))

    def test_pure_tone_not_called_band_limited(self):
        t = np.arange(44100 * 2) / 44100
        r = ip.analyze_audio((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 44100)
        self.assertFalse(any("band-limited" in f["text"] for f in r["flags"]))

    def test_viz_arrays_present(self):
        r = ip.analyze_audio(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1, 16000)
        for k in ("envelope", "spec_db", "spec_freqs", "psd_db", "frame_rms_db"):
            self.assertIn(k, r["viz"])
        self.assertEqual(r["viz"]["spec_db"].shape[0], len(r["viz"]["spec_freqs"]))


class TriageTests(unittest.TestCase):
    def test_bounds_and_labels(self):
        warn = [{"level": "warn", "text": "x"}] * 20
        t = triage.triage_score(image_flags=warn, audio_flags=warn, temporal_flags=["a"] * 9)
        self.assertLessEqual(t["score"], 100)
        self.assertEqual(t["label"], "High")
        low = triage.triage_score(image_flags=[{"level": "info", "text": "x"}])
        self.assertEqual((low["score"], low["label"]), (0.0, "Low"))

    def test_breakdown_sums_to_score_and_reports_coverage(self):
        t = triage.triage_score(audio_flags=[{"level": "warn", "text": "x"}, {"level": "weak", "text": "y"}])
        self.assertAlmostEqual(sum(r["points"] for r in t["breakdown"]), t["score"], places=1)
        self.assertEqual(t["coverage"], ["audio"])
        self.assertIn("image forensics", t["not_measured"])

    def test_low_plausibility_hand_adds_points(self):
        hand = {"plausibility": {"score": 40.0}}
        self.assertGreater(triage.triage_score(hands=[hand])["score"], 0)
        self.assertEqual(triage.triage_score(hands=[{"plausibility": {"score": 95.0}}])["score"], 0)


class CustodyTests(unittest.TestCase):
    def make(self, n=4):
        ch = []
        for i in range(n):
            custody.append(ch, "image", f"f{i}.jpg", f"{i:064x}", f"entry {i}")
        return ch

    def test_valid_chain(self):
        self.assertTrue(custody.verify(self.make())[0])
        self.assertTrue(custody.verify([])[0])

    def test_edit_delete_reorder_detected(self):
        ch = self.make()
        e = copy.deepcopy(ch); e[1]["summary"] = "tampered"
        self.assertEqual(custody.verify(e)[:2], (False, 1))
        d = copy.deepcopy(ch); del d[1]
        self.assertFalse(custody.verify(d)[0])
        r = copy.deepcopy(ch); r[1], r[2] = r[2], r[1]
        self.assertFalse(custody.verify(r)[0])
        t = copy.deepcopy(ch); t[-1]["file_sha256"] = "f" * 64
        self.assertFalse(custody.verify(t)[0])

    def test_export_roundtrip_and_tamper(self):
        ch = self.make()
        txt = custody.export_json(ch)
        self.assertTrue(custody.verify_export(txt)[0])
        doc = json.loads(txt); doc["entries"][0]["source"] = "evil.jpg"
        self.assertFalse(custody.verify_export(json.dumps(doc))[0])
        doc2 = json.loads(txt); doc2["head_hash"] = "0" * 64
        self.assertFalse(custody.verify_export(json.dumps(doc2))[0])
        self.assertFalse(custody.verify_export("not json")[0])


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        base = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        cls.a = np.asarray(Image.fromarray(base).resize((400, 300), Image.BICUBIC))

    def test_identical(self):
        r = compare.compare_images(self.a, self.a)
        self.assertGreater(r["ssim"], 0.999)
        self.assertEqual(r["changed_pixels_pct"], 0.0)
        self.assertIn("Near-identical", r["verdict"])

    def test_local_edit_is_localised(self):
        b = self.a.copy(); b[100:180, 150:260] = 255
        r = compare.compare_images(self.a, b)
        self.assertTrue(r["regions"])
        x, y = r["regions"][0]["x"], r["regions"][0]["y"]
        self.assertTrue(120 < x < 190 and 70 < y < 130, (x, y))
        self.assertIn("Localised", r["verdict"])
        for k in ("heatmap", "overlay"):
            self.assertEqual(r[k].dtype, np.uint8)

    def test_different_size_is_resized(self):
        small = np.asarray(Image.fromarray(self.a).resize((200, 150)))
        self.assertTrue(compare.compare_images(self.a, small)["second_image_resized"])


class ReportTests(unittest.TestCase):
    def test_input_report_contains_dossier_and_escapes(self):
        p = ip.profile_upload("<b>x</b>.wav", demo.demo_voice_wav())
        t = triage.triage_score(audio_flags=p["audio"]["flags"])
        page = report.build_input_report(ip.export_profile(p), t)
        self.assertIn("Input dossier", page); self.assertIn("Audio measurements", page); self.assertIn("triage", page.lower())
        self.assertNotIn("<b>x</b>.wav", page)

    def test_image_report_includes_dossier_section(self):
        from labs import image_metrics as im
        rgb = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
        raw = _png(rgb)
        res = im.analyze_image(rgb, raw, "a.png")
        prof = ip.export_profile(ip.profile_upload("a.png", raw))
        rec = report.build_record("a.png", res, [], [], [], {}, [], dossier=prof, triage=triage.triage_score(image_flags=res["flags"]))
        page = report.build_html_report(rec, res["maps"])
        self.assertIn("Input dossier", page); self.assertIn("Review-priority triage", page)


if __name__ == "__main__":
    unittest.main()
