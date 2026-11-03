# Evaluation report (honest numbers)

Setup: 14 real photos bundled offline (skimage/sklearn), 36 random 256 px crops each = 504 REAL samples;
504 FAKE samples made by our own simulators (upsampling, GAN-style grid, over-smoothing, splice/face-swap-like blend).
Accuracy is **leave-one-source-photo-out** (the test photo is never seen in training). Reproduce:
`cd evaluation && python build2.py && python evaluate2.py`

| Detector | Accuracy (balanced 50/50) | Real kept | Fakes caught |
|---|---|---|---|
| Shipped rule flags (>=1 warning) | 51.3% | 100% | 2.6% |
| Logistic regression on forensic features | 77.8% | 73.6% | 81.9% |
| Gradient boosting on forensic features | 73.7% | 60.3% | 87.1% |
| Same, fake family never seen in training | 62-69% | | |

Not measured: PyTorch EfficientNet/WavLM pipeline (no checkpoints, no torch here), real deepfakes, audio, video.
Synthetic fakes here are our own artifact simulators, so even these numbers do not transfer to real deepfakes.
Use `evaluation/train_on_folder.py` with a real labelled dataset to get a number worth quoting.
