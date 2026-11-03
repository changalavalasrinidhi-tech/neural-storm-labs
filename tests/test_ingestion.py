"""Regression: long videos must be sampled, not fully decoded (the original code wrote ~30 PNGs per second)."""
import importlib.util, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ns_ingestion", ROOT / "core" / "ingestion.py")  # avoids importing torch via core/__init__
ing = importlib.util.module_from_spec(spec); sys.modules["ns_ingestion"] = ing; spec.loader.exec_module(ing)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class VideoSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.video = cls.tmp / "v.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
                        "-t", "30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(cls.video)], check=True)

    def test_bounded_frames_time_and_disk(self):
        i = ing.MediaIngestor(temp_dir=self.tmp, max_frames=12)
        job = self.tmp / "job"; job.mkdir()
        t = time.time(); frames = i._extract_video_frames(self.video, job)
        self.assertEqual(len(frames), 12)
        self.assertLess(time.time() - t, 40)                      # original: minutes for a long clip
        self.assertEqual(list(job.glob("*.png")), [])             # no per-frame files written
        self.assertLessEqual(max(max(fr.shape[:2]) for fr in frames), 1280 + 1)          # big frames are downscaled

    def test_frames_span_whole_video(self):
        i = ing.MediaIngestor(temp_dir=self.tmp, max_frames=8)
        frames = i._extract_video_frames(self.video, self.tmp)
        self.assertGreater(abs(float(frames[0].mean()) - float(frames[-1].mean())) + abs(frames[0][:50, :50].mean() - frames[-1][:50, :50].mean()), 0.0)

    def test_corrupt_file_returns_empty_not_hang(self):
        bad = self.tmp / "bad.mp4"; bad.write_bytes(b"not a video" * 50)
        t = time.time(); self.assertEqual(ing.MediaIngestor(temp_dir=self.tmp, max_frames=8)._extract_video_frames(bad, self.tmp), [])
        self.assertLess(time.time() - t, 10)

    def test_ffmpeg_fallback(self):
        i = ing.MediaIngestor(temp_dir=self.tmp, max_frames=6); i._sample_frames_opencv = lambda src: []
        job = self.tmp / "job2"; job.mkdir()
        self.assertEqual(len(i._extract_video_frames(self.video, job)), 6)


if __name__ == "__main__":
    unittest.main()
