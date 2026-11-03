"""Unit tests for the torch-free Labs package. Run:  python -m unittest discover -s tests -v"""

import io
import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs import explorer, hand_analytics as ha, image_metrics as im, liveness, report  # noqa: E402
from labs.landmark_backend import DetectionResult, HandDet, LandmarkBackend, draw_detections  # noqa: E402
from labs.synthetic import GESTURES, make_synthetic_hand  # noqa: E402


def sample_photo() -> np.ndarray:
    try:
        from skimage import data
        return data.astronaut()
    except Exception:  # scikit-image not installed: synthetic textured image
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        return np.asarray(Image.fromarray(base).resize((512, 512), Image.BICUBIC))


def jpeg_bytes(rgb, q=85) -> bytes:
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=q); return b.getvalue()


class ImageMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb = sample_photo()
        cls.jpg = jpeg_bytes(cls.rgb, 85)
        cls.dec = np.asarray(Image.open(io.BytesIO(cls.jpg)).convert("RGB"))
        cls.res = im.analyze_image(cls.dec, cls.jpg, "t.jpg")

    def test_structure_and_json(self):
        m = self.res["metrics"]
        for k in ("basic", "hashes", "exposure", "sharpness", "noise", "spectrum", "ela", "noise_residual", "metadata", "flags"):
            self.assertIn(k, m)
        json.dumps(im.to_jsonable(m))  # must not raise
        for k, v in self.res["maps"].items():
            self.assertEqual(v.dtype, np.uint8, k)
            self.assertEqual(v.shape[-1], 3, k)

    def test_jpeg_quality_estimate_exact(self):
        for q in (30, 60, 85, 95):
            est = im.estimate_jpeg_quality(Image.open(io.BytesIO(jpeg_bytes(self.rgb, q))))
            self.assertEqual(est, q)

    def test_png_generator_metadata(self):
        info = PngImagePlugin.PngInfo(); info.add_text("parameters", "Steps: 20, Model: sdxl")
        b = io.BytesIO(); Image.fromarray(self.rgb).save(b, "PNG", pnginfo=info)
        meta = im.metadata_report(b.getvalue())
        self.assertIn("parameters", meta["generator_hits"])
        self.assertIn("sdxl", meta["generator_hits"])

    def test_noise_orders_correctly(self):
        rng = np.random.default_rng(1)
        flat = np.full((200, 200), 128, np.float32)
        noisy = flat + rng.normal(0, 10, flat.shape).astype(np.float32)
        self.assertLess(im.noise_sigma(flat), 0.5)
        self.assertAlmostEqual(im.noise_sigma(noisy), 10, delta=1.5)

    def test_sharpness_orders_correctly(self):
        g = im._gray_f32(self.dec)
        blurred = __import__("cv2").GaussianBlur(g, (0, 0), 4)
        self.assertGreater(im.sharpness_metrics(g)["laplacian_variance"], im.sharpness_metrics(blurred)["laplacian_variance"])

    def test_hash_stability(self):
        a = im.perceptual_hashes(im._gray_f32(self.dec).astype(np.uint8))["dhash"]
        small = __import__("cv2").resize(self.dec, (256, 256), interpolation=__import__("cv2").INTER_AREA)
        b = im.perceptual_hashes(im._gray_f32(small).astype(np.uint8))["dhash"]
        self.assertLessEqual(im.hamming_distance_hex(a, b), 8)

    def test_edge_cases(self):
        for arr in (np.zeros((20, 20), np.uint8), np.random.randint(0, 255, (300, 300, 4), dtype=np.uint8),
                    self.dec.astype(np.float32) / 255.0, np.zeros((5, 5, 3), np.uint8)):
            r = im.analyze_image(arr)
            json.dumps(im.to_jsonable(r["metrics"]))

    def test_large_image_is_downscaled_not_rejected(self):
        big = np.random.randint(0, 255, (2600, 2600, 3), dtype=np.uint8)
        r = im.analyze_image(big)
        self.assertEqual(r["metrics"]["basic"]["analysis_downscaled_from"], [2600, 2600])

    def test_face_consistency(self):
        out = im.face_region_consistency(self.rgb, (177, 66, 272, 161))
        self.assertTrue(out["available"])
        self.assertFalse(im.face_region_consistency(self.rgb, (0, 0, 10, 10))["available"])


class TemporalTests(unittest.TestCase):
    def test_smooth_vs_flicker(self):
        rgb = sample_photo()
        smooth = [np.clip(rgb.astype(np.int16) + i, 0, 255).astype(np.uint8) for i in range(12)]
        rng = np.random.default_rng(0)
        bad = [np.clip(rgb.astype(np.int16) + int(rng.integers(-40, 40)), 0, 255).astype(np.uint8) for _ in range(12)]
        self.assertEqual(im.temporal_series(smooth)["flags"], [])
        self.assertTrue(any("flicker" in f.lower() for f in im.temporal_series(bad)["flags"]))
        self.assertFalse(im.temporal_series([])["available"])


class HandTests(unittest.TestCase):
    W, H = 640, 480
    EXPECT = {"open": ("Open palm", 5), "fist": ("Fist", 0), "point": ("Pointing", 1), "peace": ("Peace / victory", 2),
              "three": ("Three", 3), "four": ("Four", 4), "rock": ("Rock", 2), "thumbs_up": ("Thumbs up", 1)}

    def hand(self, g, **kw):
        return ha.analyze_hand(make_synthetic_hand(g, self.W, self.H, **kw), self.W, self.H, "Right", 0.95)

    def test_gestures(self):
        for g in GESTURES:
            m = self.hand(g)
            self.assertEqual((m["gesture"], m["finger_count"]), self.EXPECT[g], g)

    def test_mirrored_left_hand(self):
        for g in GESTURES:
            m = self.hand(g, left_hand=True)
            self.assertEqual(m["finger_count"], self.EXPECT[g][1], g)

    def test_rotation_invariance_of_finger_count(self):
        for g in ("open", "peace", "fist", "three"):
            lm = make_synthetic_hand(g, self.W, self.H)
            px = ha.to_pixel_space(lm, self.W, self.H)
            c = px[:, :2].mean(0)
            for deg in (30, 90, 180, 270):
                t = np.radians(deg); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                rot = px.copy(); rot[:, :2] = (px[:, :2] - c) @ R.T + c
                norm = np.stack([rot[:, 0] / self.W, rot[:, 1] / self.H, rot[:, 2] / self.W], 1)
                m = ha.analyze_hand(norm, self.W, self.H, "Right", 0.95)
                self.assertEqual(m["finger_count"], self.EXPECT[g][1], f"{g}@{deg}")

    def test_scale_invariance(self):
        a = self.hand("peace", palm_px=100)["finger_count"]
        b = self.hand("peace", palm_px=220)["finger_count"]
        self.assertEqual(a, b)

    def test_plausibility_drops_for_malformed_hand(self):
        lm = make_synthetic_hand("open", self.W, self.H)
        good = ha.analyze_hand(lm, self.W, self.H, "Right", 0.95)["plausibility"]["score"]
        bad = lm.copy()
        for i in (18, 19, 20):               # pinky stretched far beyond the middle finger
            bad[i, 1] = bad[17, 1] - (bad[17, 1] - bad[i, 1]) * 2.6
        for i in (14, 15, 16):               # ring finger shrunk
            bad[i, 1] = bad[13, 1] - (bad[13, 1] - bad[i, 1]) * 0.45
        score = ha.analyze_hand(bad, self.W, self.H, "Right", 0.95)["plausibility"]
        self.assertGreater(good, 90)
        self.assertLess(score["score"], good - 15)
        self.assertGreaterEqual(score["n_warn"] + score["n_fail"], 2)

    def test_low_confidence_and_cropping_penalised(self):
        lm = make_synthetic_hand("open", self.W, self.H, center=(0.95, 0.5))
        m = ha.analyze_hand(lm, self.W, self.H, "Right", 0.4)
        statuses = {c["check"]: c["status"] for c in m["plausibility"]["checks"]}
        self.assertEqual(statuses["Detector confidence"], "fail")
        self.assertIn(statuses["Landmarks inside frame"], ("warn", "fail"))

    def test_scene_checks(self):
        hands = [self.hand("open") for _ in range(3)]
        notes = ha.scene_checks(hands, n_faces=1)
        self.assertTrue(any(n["level"] == "warn" for n in notes))
        self.assertEqual(ha.scene_checks([], 0)[0]["text"], "No hands detected.")

    def test_metrics_are_json_serialisable(self):
        json.dumps(im.to_jsonable(self.hand("peace")))


class LivenessTests(unittest.TestCase):
    def test_pass_fail_expiry_replay(self):
        hands = [ha.analyze_hand(make_synthetic_hand("peace", 640, 480), 640, 480, "Right", 0.95)]
        now = 1000.0
        ch = {"id": "x", "kind": "fingers", "target": 2, "issued_at": now, "ttl": 90, "prompt": ""}
        self.assertTrue(liveness.verify(ch, hands, now=now + 5)["passed"])
        self.assertFalse(liveness.verify({**ch, "target": 4}, hands, now=now + 5)["passed"])
        self.assertIn("expired", liveness.verify(ch, hands, now=now + 500)["reason"])
        self.assertIn("replay", liveness.verify(ch, hands, frame_sha="abc", used_hashes={"abc"}, now=now + 5)["reason"])
        self.assertFalse(liveness.verify(ch, [], now=now + 5)["passed"])
        g = {"id": "y", "kind": "gesture", "target": "Peace / victory", "issued_at": now, "ttl": 90, "prompt": ""}
        self.assertTrue(liveness.verify(g, hands, now=now + 1)["passed"])

    def test_low_score_hand_rejected(self):
        h = ha.analyze_hand(make_synthetic_hand("peace", 640, 480), 640, 480, "Right", 0.3)
        ch = {"id": "x", "kind": "fingers", "target": 2, "issued_at": 0.0, "ttl": 90, "prompt": ""}
        self.assertFalse(liveness.verify(ch, [h], now=1.0)["passed"])

    def test_new_challenge_shape(self):
        for _ in range(50):
            c = liveness.new_challenge()
            self.assertIn(c["kind"], ("fingers", "gesture"))
            if c["kind"] == "fingers":
                self.assertIn(c["target"], range(1, 6))
            else:
                self.assertIn(c["target"], liveness.GESTURE_TARGETS)


class BackendTests(unittest.TestCase):
    def test_detect_finds_face_and_never_raises(self):
        b = LandmarkBackend(ROOT / "models")
        st = b.status()
        self.assertIn(st["mode"], ("solutions", "tasks", "opencv-haar"))
        rgb = sample_photo()
        for arr in (rgb, rgb[..., 0], np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), rgb.astype(np.float32) / 255, rgb[:30, :30]):
            r = b.detect(arr)
            self.assertIsInstance(r, DetectionResult)
        r = b.detect(rgb)
        if r.faces_available:
            self.assertGreaterEqual(len(r.faces), 1)
        if st["mode"] != "solutions" and not st["hands_ready"]:
            self.assertTrue(any("hand" in n.lower() for n in r.notes))

    def test_draw_does_not_mutate_input(self):
        rgb = sample_photo(); before = rgb.copy()
        r = DetectionResult(hands=[HandDet(make_synthetic_hand("open", 512, 512), "Left", 0.9)])
        out = draw_detections(rgb, r)
        self.assertTrue(np.array_equal(rgb, before))
        self.assertFalse(np.array_equal(out, rgb))


class ReportExplorerTests(unittest.TestCase):
    def test_report_roundtrip(self):
        rgb = sample_photo(); jpg = jpeg_bytes(rgb)
        res = im.analyze_image(np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")), jpg, "a.jpg")
        hand = ha.analyze_hand(make_synthetic_hand("peace", 512, 512), 512, 512, "Right", 0.9)
        rec = report.build_record("a.jpg", res, [hand], [{"box": [1, 2, 3, 4], "score": None}], ha.scene_checks([hand], 1),
                                  {"mode": "test"}, ["note"])
        json.dumps(rec)
        page = report.build_html_report(rec, res["maps"])
        for needle in ("evidence report", "Skeleton plausibility", "Peace / victory", "data:image/png;base64", "Error Level Analysis"):
            self.assertIn(needle, page)
        row = report.history_row(rec)
        self.assertEqual(row["hands"], 1)

    def test_html_escapes_filename(self):
        rgb = sample_photo()
        res = im.analyze_image(rgb)
        rec = report.build_record("<script>alert(1)</script>", res, [], [], [], {}, [])
        self.assertNotIn("<script>alert(1)</script>", report.build_html_report(rec, {}))

    def test_explorer(self):
        rows = explorer.scan_project(ROOT)
        paths = {r["path"] for r in rows}
        for p in ("app.py", "config.py", "labs/image_metrics.py", "core/fusion.py"):
            self.assertIn(p, paths)
        self.assertFalse(any("__pycache__" in p for p in paths))
        self.assertIn("def main", explorer.read_text(ROOT, "app.py"))
        with self.assertRaises(ValueError):
            explorer.read_text(ROOT, "../../etc/passwd")
        self.assertTrue(len(explorer.health_check(ROOT)) > 5)


if __name__ == "__main__":
    unittest.main()
