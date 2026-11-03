"""UI smoke test with a fake streamlit:  python tests/smoke_ui.py"""
import importlib, io, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
from PIL import Image
import fake_streamlit as fs

def load():
    for m in [m for m in sys.modules if m in ("app", "ui_labs")]: del sys.modules[m]
    st = fs.install()
    return st, importlib.import_module("app")

def photo_bytes(fmt="JPEG"):
    try:
        from skimage import data; rgb = data.astronaut()
    except Exception:
        rgb = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    b = io.BytesIO(); Image.fromarray(rgb).save(b, fmt); return b.getvalue()

def run():
    results = []
    def check(name, cond): results.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)

    # 1. no upload
    st, app = load(); app.main()
    check("landing renders with no upload", any("Start an investigation" in t for t in st.texts("markdown")))
    check("no st.error on empty state", not st.texts("error"))
    check("all 5 tabs rendered (explorer table + research glossary)", st.texts("dataframe") and any("Research notes" in t for t in st.texts("markdown")))
    check("core-missing warning shown when torch absent" , app.CORE_OK or st.texts("warning"))

    # 2. image upload
    st, app = load(); st.file_upload = fs.FakeUpload("sample.jpg", photo_bytes(), "image/jpeg"); app.main()
    check("image upload: no errors", not st.texts("error"))
    check("image lab rendered metrics", any(c[0] == "metric" for c in st.calls))
    check("image lab export buttons", len(st.texts("download_button")) >= 3)
    check("session log got a row", len(st.session_state.get("ns_history", [])) == 1)
    app.main()  # rerun must not duplicate or recompute
    check("rerun does not duplicate log rows", len(st.session_state["ns_history"]) == 1)

    # 3. camera + liveness
    st, app = load(); st.camera = fs.FakeUpload("c.png", photo_bytes("PNG"), "image/png")
    st.buttons_pressed = {"new_challenge"}; app.main()
    check("camera: no errors", not st.texts("error"))
    ch = st.session_state.get("ns_challenge"); check("challenge issued", ch is not None)
    st.buttons_pressed = {"verify_challenge"}; app.main()
    res = st.session_state.get("ns_challenge_result"); check("verify returns a result (no hands in photo => not passed, no crash)", res is not None and res["passed"] is False)

    # 4. broken audio upload does not crash and says so
    st, app = load(); st.file_upload = fs.FakeUpload("x.wav", b"RIFF....", "audio/wav"); app.main()
    check("corrupt audio: no uncaught crash", not st.texts("error"))
    check("corrupt audio: user is told it could not be measured", any("could not be measured" in t for t in st.texts("warning")))

    # 5. real audio: banner, dossier, audio lab, custody chain
    from labs import demo
    st, app = load(); st.file_upload = fs.FakeUpload("voice.wav", demo.demo_voice_wav(), "audio/wav"); app.main()
    check("audio upload: no errors", not st.texts("error"))
    check("audio upload: INPUT RECEIVED banner rendered", any("INPUT RECEIVED" in t and "AUDIO" in t for t in st.texts("markdown")))
    check("audio upload: audio lab heading", any("Audio lab" in t for t in st.texts("markdown")))
    check("audio upload: dossier heading", any("Input dossier" in t for t in st.texts("markdown")))
    check("audio upload: audio metrics shown", sum(c[0] == "metric" for c in st.calls) >= 12)
    check("audio upload: chain has 1 verified entry", len(st.session_state.get("ns_chain", [])) == 1)
    app.main()
    check("audio rerun does not duplicate chain/log", len(st.session_state["ns_chain"]) == 1 and len(st.session_state["ns_history"]) == 1)

    # 6. demo mode (no upload)
    st, app = load(); st.session_state["ns_demo"] = True; app.main()
    check("demo mode renders audio lab without an upload", not st.texts("error") and any("Audio lab" in t for t in st.texts("markdown")))
    st, app = load(); st.buttons_pressed = {"load_demo"}; app.main()
    check("landing demo button sets demo flag", st.session_state.get("ns_demo") is True)

    # 7. compare tab
    st, app = load(); b = photo_bytes("PNG")
    st.keyed_uploads = {"cmp_a": fs.FakeUpload("a.png", b, "image/png"), "cmp_b": fs.FakeUpload("b.png", b, "image/png")}; app.main()
    check("compare tab: identical images, no errors", not st.texts("error") and len(st.session_state.get("ns_cmp", {})) == 1)

    # 8. microphone clip
    st, app = load(); st.mic = fs.FakeUpload("mic.wav", demo.demo_voice_wav(), "audio/wav"); app.main()
    check("microphone clip: audio lab rendered, no errors", not st.texts("error") and any("Audio lab" in t for t in st.texts("markdown")))

    # 9. video with an audio track (needs ffmpeg to build the sample)
    import shutil, subprocess, tempfile
    if shutil.which("ffmpeg"):
        out = Path(tempfile.mkdtemp()) / "v.mp4"
        subprocess.run(f"ffmpeg -v error -y -f lavfi -i testsrc=duration=2:size=160x120:rate=10 -f lavfi -i sine=frequency=300:duration=2 -c:v mpeg4 -c:a aac {out}", shell=True)
        st, app = load(); st.file_upload = fs.FakeUpload("clip.mp4", out.read_bytes(), "video/mp4"); app.main()
        check("video upload: no errors", not st.texts("error"))
        check("video upload: banner says VIDEO and audio lab + frame lab shown",
              any("INPUT RECEIVED" in t and "VIDEO" in t for t in st.texts("markdown")) and any("Audio lab" in t for t in st.texts("markdown"))
              and any("Video frame lab" in t for t in st.texts("markdown")))

    print(f"\n{sum(ok for _, ok in results)}/{len(results)} smoke checks passed")
    sys.exit(0 if all(ok for _, ok in results) else 1)

if __name__ == "__main__":
    run()
