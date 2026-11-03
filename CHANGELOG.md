# Changelog

## v2 (Labs)
**New** – `labs/` package (no PyTorch needed): image metrics (sharpness, noise, entropy, blockiness, FFT spectrum, ELA, noise residual, JPEG quality, EXIF, generator/C2PA markers, perceptual hashes), hand analytics (joint angles, proportions, gesture, skeleton plausibility), face-vs-surroundings consistency, video temporal consistency, gesture challenge, HTML/JSON evidence report, project explorer + health check, research tab.
**Fixed**
- `mp.solutions` was assumed; newer mediapipe removed it, so face/hand overlay silently failed. Now supports both APIs.
- `requirements.txt` pins (`mediapipe==0.10.21`, `numpy<2`) are not installable on Python 3.13; relaxed.
- `_save_upload` leaked a file descriptor per upload; temp files/job folders were never deleted.
- The whole pipeline re-ran on every widget click; results are now cached per file+settings.
- `use_container_width` is renamed in newer Streamlit; compatibility wrappers added.
- App no longer dies when PyTorch is missing; the Labs tabs keep working.
- Removed build artefacts (`__pycache__`, `ARCHITECTURE.md.txt` – a PowerShell scaffold duplicating `ARCHITECTURE.md`).

## Verified vs. not verified
Verified in a sandbox without PyTorch/Streamlit: all `labs` logic (unit tests), UI code paths (fake streamlit), pipeline wiring (stubbed).
**Not verified**: the PyTorch pipeline itself, real Streamlit rendering, real hand detection (no model files / network in the sandbox), thresholds on real hands (tuned on synthetic skeletons).

## v2.1 – large-video hang fix
- **Fixed:** uploading a long video (e.g. 2:38, 157 MB) froze at "1/6 Inspecting media format". `core/ingestion.py` decoded *every* frame at 30 FPS to PNG and held them all in RAM before keeping 16. Measured on a 20 s 1080p clip: 92 s and 3.6 GB RAM, 600 files; the same clip now takes ~5 s, 169 MB, 0 files. A 160 s / 310 MB clip: ~15 s. The OpenCV fallback had the same flaw.
- Frames are now sampled evenly across the whole video (seek-based), downscaled to ≤1280 px, with bounded ffmpeg and sequential fallbacks; only the first 12 s of audio is decoded.
- Streamlit upload limit raised to 1 GB.
- Left column under the preview now shows a 'What was analysed' summary: frames, faces found, audio track, and per-stream status with the reason a stream did not run.

## v2.2 - input dossier, audio lab and hackathon extras
- **Added:** `INPUT RECEIVED` banner + **Input dossier** for every submission (image / video / audio / camera / mic): detected type, size, SHA-256, resolution, fps, duration, codec, audio-stream facts, and a per-analysis "runs / does not run, and why" plan. JSON export included.
- **Added:** **Audio lab** (`labs/input_profile.py`, torch-free): works for audio files, a video's audio track and microphone clips; waveform, spectrogram, spectrum, loudness, clipping, bandwidth cut-off, pitch, noise floor, digital-silence gaps, level jumps; HTML export.
- **Added:** review-priority **triage** score (`labs/triage.py`) with transparent breakdown and coverage; included in image, audio and video views and in exports.
- **Added:** tamper-evident **chain of custody** (`labs/custody.py`) in the Session log tab with export and re-verification.
- **Added:** **Tamper Compare** tab (`labs/compare.py`), **microphone clip** input, **demo mode** (`labs/demo.py`), accepted-input legend in the sidebar and landing page.
- **Changed:** session-log CSV now supports rows of different media kinds; HTML evidence report gained dossier + triage sections.
- **Tests:** 55 unit tests (30 new) and 25 UI smoke checks, including real WAV/MP3/MP4 (with and without audio) generated with ffmpeg.
- **Not verified here:** real Streamlit rendering (tested with a fake streamlit), the PyTorch pipeline, and `st.audio_input` on a live browser (needs Streamlit >= 1.39; the app explains this if it is missing). Audio indicators are heuristics tuned on synthetic and ffmpeg-generated signals, not on a labelled deepfake-audio corpus.
