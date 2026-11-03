# Project Specification: Neural Storm FT-02 Deepfake & Synthetic Media Detection Engine

## Overview
Neural Storm FT-02 is a software-only, multi-modal deepfake detection platform capable of analyzing Images, Videos, and Audio files. It features a universal input router, dual-domain visual detection, speech vocoder analysis, lip-sync verification, score calibration with uncertainty estimation, and interactive Grad-CAM++ explainability within a Streamlit UI.

## Tech Stack
- UI & Orchestration: Streamlit
- Computer Vision: PyTorch, OpenCV, MediaPipe, `timm` (EfficientNet-B4), torchvision
- Audio Processing: Torchaudio, Librosa, FFmpeg
- ML Models: EfficientNet-B4 (Spatial RGB), High-Pass DCT Residual Filter (Frequency), WavLM / AASIST (Audio), SyncNet (Lip-sync)

## Project Directory Layout
neural_storm_ft02/
├── app.py                      # Main Streamlit UI & Orchestrator
├── config.py                   # Central paths, weights, and confidence thresholds
├── requirements.txt            # Project dependencies
├── ARCHITECTURE.md             # Architecture reference
├── core/
│   ├── __init__.py
│   ├── ingestion.py            # Universal media validation, FFmpeg demux, MediaPipe face cropping
│   ├── visual_detector.py      # Dual-domain (EfficientNet-B4 + DCT Residuals) classifier
│   ├── audio_detector.py       # WavLM / AASIST neural vocoder detector
│   ├── sync_engine.py          # SyncNet phoneme-viseme cosine distance engine
│   ├── fusion.py               # Dynamic quality-weighted fusion & temperature calibration
│   └── explainability.py       # Grad-CAM++ spatial heatmaps & spectrogram highlighting
└── models/                     # Model weights directory (.pt / .onnx)

## Universal Media Flow Strategy
1. Image Input: Ingest -> MediaPipe Face Crop -> Visual Stream (RGB + DCT) -> Grad-CAM++ Heatmap -> Output.
2. Audio Input: Ingest -> Torchaudio 16kHz Resample -> Audio Stream (WavLM/AASIST) -> Spectrogram Highlight -> Output.
3. Video Input: Ingest -> FFmpeg Demux -> Visual Stream + Audio Stream + SyncNet Lip-Sync Check -> Calibrated Score Fusion -> Unified Diagnostics.

## Fusion Formula
Logit_fused = w_v * z_visual + w_a * z_audio + w_s * z_sync
Scores are calibrated via Temperature Scaling. If Epistemic Uncertainty > Threshold (or 0.40 <= Score <= 0.60), flag as "Uncertain / Requires Forensic Review".
