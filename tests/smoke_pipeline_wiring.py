"""Checks the app's pipeline wiring (caching, unpacking, cleanup) with a stubbed run_pipeline."""
import sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import smoke_ui as su
fs = su.fs

st, app = su.load()
calls = {"n": 0, "paths": []}
SN = types.SimpleNamespace
def fake_run(path, engines, sens, progress, status_box):
    calls["n"] += 1; calls["paths"].append(path); assert Path(path).is_file()
    media = SN(media_type="image", mime_type="image/jpeg", faces=[1], frames=[1], has_audio=False, metadata={}, face_boxes=[])
    fusion = SN(probability_pct=61.0, label="Uncertain / Requires Forensic Review", confidence_interval=(.5, .7), epistemic_uncertainty=.1,
                fused_logit=.3, weights={"w_v": 1., "w_a": 0., "w_s": 0.}, probability=.61, sqi={}, stream_logits={}, stream_probs={}, metadata={})
    bundle = SN(original_face=None, overlay_rgb=None, heatmap_rgb=None, spectrogram_overlay=None, spectrogram_raw=None, narrative="n")
    return media, {"using_fallback_weights": True}, None, {"active": False}, fusion, bundle
app.CORE_OK = True
app.load_engines = lambda d: {"device": "cpu", "status": {"visual": "fallback"}}
app.run_pipeline = fake_run
st.file_upload = fs.FakeUpload("sample.jpg", su.photo_bytes(), "image/jpeg")
app.main(); app.main()
ok = []
ok.append(("pipeline ran exactly once across two reruns (cache works)", calls["n"] == 1))
ok.append(("temp upload file cleaned up", not Path(calls["paths"][0]).exists()))
ok.append(("verdict rendered", any("Calibrated verdict" in t for t in st.texts("subheader"))))
ok.append(("fallback warning shown", any("heuristic demo" in t for t in st.texts("caption"))))
ok.append(("summary fills left column", any("What was analysed" in x for x in st.texts("markdown"))))
ok.append(("no-face notice shown", any("No face was detected" in x for x in st.texts("info"))))
ok.append(("no errors", not st.texts("error")))
st.file_upload = fs.FakeUpload("other.jpg", su.photo_bytes("PNG"), "image/jpeg"); app.main()
ok.append(("new file triggers a fresh run", calls["n"] == 2))
for n, c in ok: print(("PASS " if c else "FAIL ") + n)
sys.exit(0 if all(c for _, c in ok) else 1)
