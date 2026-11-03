# Neural Storm FT-02 — UI redesign notes

## What changed
- Reworked the Streamlit visual theme into a polished dark research-studio interface with indigo, mint, and violet accents, stronger typography, glass-like panels, improved metrics, and a clearer landing state.
- Added an optional **Face + hand landmark overlay** for uploaded images using MediaPipe. It draws face bounding boxes and hand skeleton landmarks and reports the counts found.
- Improved the analysis workspace copy and evidence section hierarchy for a hackathon presentation.
- Added a clear research-prototype note: visual landmarks are not identity recognition, and model outputs are not proof of authenticity/deception.

## Important model note
The original project still depends on model checkpoints in `models/` for trained detector performance. If those checkpoints are absent, existing fallback behavior remains. The new face/hand overlay uses MediaPipe's built-in landmark models when installed; it is not an object detector for arbitrary objects. The existing visual/audio/fusion pipeline remains the core media-forensics analysis.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
