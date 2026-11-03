# Neural Storm FT-02 · Labs v2

Deepfake / synthetic-media forensic workspace (Streamlit) with a new torch-free **Labs** layer.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
If your `mediapipe` is newer than 0.10.21 (no `mp.solutions`), hand/face landmarks need two small model files:
```bash
python scripts/download_models.py     # or press "Download MediaPipe models" in the Camera tab
```
Without them faces still work (OpenCV Haar) and the app says clearly that hands are unavailable.

## What was submitted? (new in v2.2)
Every input - image, video, audio, camera snapshot or microphone clip - opens with an **INPUT RECEIVED** banner:
the detected type, file name, size, resolution / fps / duration / sample rate, whether a video has an audio track,
and which analyses will run (and, for those that cannot, the reason). The **Input dossier** section below lists
every file, media and audio-stream fact and exports it as JSON.

## Tabs
| Tab | What it does |
|---|---|
| Forensic Studio | Input banner + dossier, original pipeline (visual / audio / sync / fusion / Grad-CAM++), **Audio lab** (audio files and video audio tracks), image & hand lab for images, frame lab + temporal consistency for video, **review-priority triage** |
| Camera · Mic · Hand Lab | Snapshot → face/hand detection, all metrics, gesture challenge (experimental liveness); **microphone clip → Audio lab** |
| Tamper Compare | Original vs suspect image: SSIM, pixel change, difference heat-map, changed regions, perceptual-hash distance |
| Project Explorer | Every project file with size/lines/role, a file viewer, and an environment health check |
| Research | Sourced research notes and a metric glossary |
| Session log & custody | Table + CSV of everything analysed, plus a **tamper-evident SHA-256 hash chain** you can export and re-verify |

## Hackathon extras
- **Audio lab** - waveform, spectrogram, long-term spectrum, loudness, clipping, bandwidth cut-off, pitch, noise floor, digital-silence gaps, abrupt level jumps, with plain-language indicators.
- **Review-priority triage** - 0-100 score where every point is traceable to a signal, and the panel states what was *not* measured. It is not a probability of fakeness.
- **Chain of custody** - exported log can be re-verified; edits, deletions or re-ordering are detected.
- **Tamper Compare** - localises edits between two versions of an image.
- **Demo mode** - one click loads a synthetic voice clip with detectable traits, so the app can be shown with no files.

## Test
```bash
python -m unittest discover -s tests -v   # logic (55 tests)
python tests/smoke_ui.py                  # UI wiring with a fake streamlit
python tests/smoke_pipeline_wiring.py     # cache / cleanup wiring
```
All outputs are indicators, not proof. No identity recognition is performed.
