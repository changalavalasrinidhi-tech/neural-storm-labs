"""Research notes, references and a metric glossary shown in the app's Research tab.

Sources were gathered with web search on 2026-10-18. Summaries are paraphrased; open the links
for the original wording and exact claims.
"""

from __future__ import annotations

from typing import Dict, List

TOPICS: List[Dict[str, object]] = [
    {
        "title": "Hands as a forensic clue",
        "findings": [
            "Image generators have a long record of malformed hands: missing, extra or merged fingers, odd proportions and unnatural joints. A 2024 guide for telling AI images from photographs lists hands first among anatomical implausibilities, while stressing that they are neither always present nor proof on their own.",
            "A 2025 study (HandEval) built a 48k-image hand-quality dataset and argues that evaluating hand regions is an under-explored signal for AI-image detection.",
            "Hand-repair research (HandRefiner) fixes bad hands by fitting a hand-mesh model that always yields the correct finger count - which also shows why landmark/mesh models tend to 'normalise' a malformed hand instead of exposing it.",
            "Caution from fact-checking practice: a widely shared 'six-finger' frame turned out to be a palm bulge plus lighting and compression, and detector tools found no synthetic signs. One ambiguous frame is weak evidence.",
        ],
        "in_project": "Hand Lab reports finger count, joint angles, proportions and a skeleton plausibility score, plus detector confidence and hands-per-face. Because landmark models regularise shapes, low confidence, odd proportions and scene-level inconsistencies are the useful signals - treat all of it as a screening aid.",
        "refs": [
            ("How to Distinguish AI-Generated Images from Authentic Photographs (arXiv 2406.08651)", "https://arxiv.org/pdf/2406.08651"),
            ("HandEval: Hand Quality Evaluation in Generated Images (arXiv 2510.08978)", "https://arxiv.org/pdf/2510.08978"),
            ("HandRefiner: Refining Malformed Hands (arXiv 2311.17957)", "https://arxiv.org/pdf/2311.17957"),
            ("Fact-check example of an apparent 'sixth finger'", "https://lenz.io/c/a750d0e3"),
        ],
    },
    {
        "title": "MediaPipe hand and face landmarks",
        "findings": [
            "The Hand Landmarker bundles two models: a palm detector that finds hands, and a landmark model that localises 21 hand-knuckle coordinates inside each detected region. It reports image-space landmarks, world-space landmarks and handedness.",
            "In video/stream modes it tracks from the previous frame's landmarks and only re-runs palm detection when tracking fails; this project uses single images, so palm detection runs every time.",
            "Newer mediapipe releases removed the legacy `mp.solutions` API in favour of the Tasks API, which loads separate model files. This project supports both and falls back to OpenCV Haar for faces.",
        ],
        "in_project": "labs/landmark_backend.py auto-selects `solutions` or `tasks`, never crashes on missing models, and reports exactly what is unavailable.",
        "refs": [("MediaPipe Hand Landmarker guide", "https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker")],
    },
    {
        "title": "Frequency-domain artifacts of generated images",
        "findings": [
            "Durall et al. (ICML 2020) found that GAN images show clear artifacts in the Fourier spectrum and traced them to upsampling operations common to GAN architectures.",
            "Zhang et al. showed the upsampling leaves replicated spectra and trained a spectrum-based classifier that generalised to unseen GAN models.",
            "Later work on diffusion models reports differences from real images in high-frequency bands, but also notes that frequency artifacts alone generalise poorly because patterns differ between generators.",
            "Counter-point: follow-up research found that fingerprints can sometimes be removed by an adversary, so spectral cues degrade under deliberate evasion or heavy re-compression.",
        ],
        "in_project": "The spectrum panel reports high-frequency energy ratio, spectral slope and isolated periodic peaks (green dots on the spectrum). The core detector's DCT/SRM stream targets the same family of cues.",
        "refs": [
            ("Leveraging Frequency Analysis for Deep Fake Image Recognition (arXiv 2003.08685)", "https://arxiv.org/pdf/2003.08685v2"),
            ("Detecting and Simulating Artifacts in GAN Fake Images (arXiv 1907.06515)", "https://arxiv.org/pdf/1907.06515v1"),
            ("Rethinking up-sampling artifacts for generalisable detection (arXiv 2312.10461)", "https://arxiv.org/html/2312.10461v2"),
            ("Enhancing Frequency Forgery Clues for Diffusion-Generated Image Detection (arXiv 2511.00429)", "https://arxiv.org/pdf/2511.00429v1"),
        ],
    },
    {
        "title": "Error Level Analysis (ELA) - useful, easily misread",
        "findings": [
            "ELA re-saves a JPEG at a known quality and maps the difference from the original; regions at a different compression history can stand out.",
            "It cannot work meaningfully on lossless formats, scaling/recolouring raises error everywhere, and a bright area means 'higher error potential', not 'edited'. Experts have called it subjective, and a prominent forensic researcher is reported to have said it mislabels edited and original images about equally often.",
            "Skilled edits can be flattened by repeated re-saving, so a clean ELA map does not prove authenticity either.",
        ],
        "in_project": "The ELA map and tile-variation statistics are labelled as indicators, and a note appears for non-JPEG input. Use them to decide where to look, never as a verdict.",
        "refs": [
            ("Error level analysis (Wikipedia overview)", "https://en.wikipedia.org/wiki/Error_level_analysis"),
            ("FotoForensics: Common analysis mistakes", "https://fotoforensics.com/tutorial-mistakes.php"),
            ("FotoForensics ELA tutorial", "https://ipv4.fotoforensics.com/tutorial.php?tt=ela"),
            ("ELA explained and why it is weaker evidence than it looks", "https://www.ventavid.com/glossary/error-level-analysis"),
        ],
    },
    {
        "title": "Provenance and metadata",
        "findings": [
            "Some local generation tools write their settings into PNG text chunks (for example a 'parameters' or 'prompt' entry); finding one is strong evidence of the *tool*, and its absence proves nothing because metadata is trivially stripped.",
            "Content Credentials (C2PA) embed signed provenance. This project only detects the presence of a marker; it does not validate signatures.",
        ],
        "in_project": "Metadata panel: EXIF summary, JPEG quality estimate from quantisation tables, generator/editor keyword hits, PNG text keys, C2PA marker presence, GPS-present flag (coordinates are never displayed or exported).",
        "refs": [],
    },
    {
        "title": "Calibration and honest uncertainty (original pipeline)",
        "findings": [
            "Modern neural networks are often over-confident; temperature scaling is a simple, effective post-hoc fix (Guo et al., 2017). The original pipeline applies temperature scaling and flags an 'Uncertain' band instead of forcing a verdict.",
            "Without trained checkpoints in models/, the visual/audio/sync detectors run on heuristic priors plus untrained heads: the output is a demo signal, not a measured accuracy.",
        ],
        "in_project": "The Forensic Studio tab keeps the original fusion/uncertainty logic and shows when it is running in fallback mode.",
        "refs": [("On Calibration of Modern Neural Networks (arXiv 1706.04599)", "https://arxiv.org/abs/1706.04599")],
    },
    {
        "title": "Gesture challenge-response (experimental)",
        "findings": [
            "Idea: ask for a random, short-lived gesture and verify it from the camera frame, so a static pre-made image or clip is less likely to match.",
            "Not validated: a real-time face/hand deepfake, or a prepared set of images, can still pass. No published validation backs this implementation; it is a research prototype only.",
        ],
        "in_project": "Camera & Hand Lab -> Liveness challenge: random finger count or gesture, 90 s expiry, replay blocking by frame hash.",
        "refs": [],
    },
    {
        "title": "Ethics and limits",
        "findings": [
            "Landmarks are geometry, not identity. Nothing here recognises a person.",
            "Every score is an indicator with known failure modes (compression, filters, lighting, small hands, unusual poses, skin-tone and camera bias in training data). Do not use any output as sole evidence about a person or a file.",
        ],
        "in_project": "Disclaimers are repeated in the UI and in every exported report.",
        "refs": [],
    },
]

METRIC_GLOSSARY: List[Dict[str, str]] = [
    {"metric": "Laplacian variance", "meaning": "Sharpness / detail. Higher = sharper.", "reading": "Resolution dependent; <40 very soft, 40-120 soft, 120-800 sharp, >800 very detailed.", "caveat": "Noise and sharpening raise it; depth of field lowers it."},
    {"metric": "Noise sigma (Immerkaer)", "meaning": "Estimated sensor-noise standard deviation in grey levels.", "reading": "Camera photos usually >1; near 0 means unusually smooth.", "caveat": "Denoise, beauty filters, screenshots and synthesis all lower it."},
    {"metric": "Entropy (bits)", "meaning": "Information in the grey histogram (max 8).", "reading": "Natural photos ~6.5-7.8.", "caveat": "Dark / flat images score low."},
    {"metric": "Blockiness ratio", "meaning": "Edge strength on 8-px block borders vs inside blocks.", "reading": "~1.0 none; >1.1 visible JPEG grid.", "caveat": "Resized images shift the grid."},
    {"metric": "High-freq energy ratio", "meaning": "Share of spectral power in the outer half of the frequency radius.", "reading": "Typically a few percent for photos.", "caveat": "Sharpening, noise and upscaling change it."},
    {"metric": "Spectral slope", "meaning": "Log-log slope of radial power vs frequency.", "reading": "Natural images ~ -2 to -3.", "caveat": "Resampling and generators bend it; not decisive."},
    {"metric": "Periodic peaks", "meaning": "Isolated bright dots in the high-frequency spectrum.", "reading": "0-3 typical; many may mean a resampling/upsampling grid.", "caveat": "JPEG blocks and moire also create peaks."},
    {"metric": "ELA mean / tile CV", "meaning": "Average re-compression error and how unevenly it is spread across 6x6 tiles.", "reading": "High CV = uneven compression history.", "caveat": "Edges and textures are naturally brighter; PNG input weak."},
    {"metric": "JPEG quality estimate", "meaning": "Quality factor inferred from the luminance quantisation table.", "reading": "Exact for libjpeg-style tables.", "caveat": "Custom camera tables give approximate values."},
    {"metric": "Perceptual hashes", "meaning": "64-bit aHash/dHash fingerprints that survive resizing.", "reading": "Hamming distance <10 means visually near-identical.", "caveat": "Cropping breaks them."},
    {"metric": "Face/ring sharpness ratio", "meaning": "Face-box Laplacian variance over the surrounding ring.", "reading": "0.4-2.5 normal.", "caveat": "Depth of field, make-up and retouching also shift it."},
    {"metric": "PIP / IP angle", "meaning": "Interior joint angle (180 = straight).", "reading": ">=150 counts as extended.", "caveat": "Foreshortened fingers are measured with noisy depth."},
    {"metric": "Curl %", "meaning": "Mean finger flexion scaled to 0-100.", "reading": "0 straight, 100 fully curled.", "caveat": "-"},
    {"metric": "Openness index", "meaning": "Mean fingertip-to-wrist distance over palm length.", "reading": "~0.7 fist, ~1.9 open hand.", "caveat": "-"},
    {"metric": "Spread angles", "meaning": "Angle between neighbouring finger directions and thumb-index.", "reading": "Adjacent fingers <=45 deg, thumb-index <=110 deg.", "caveat": "-"},
    {"metric": "Skeleton plausibility", "meaning": "Weighted pass/warn/fail of proportion, spread, confidence and framing checks.", "reading": ">=80 plausible, 55-80 some oddities, <55 implausible.", "caveat": "Landmark models regularise shapes; this misses many generator hand errors."},
    {"metric": "Detector confidence", "meaning": "MediaPipe handedness score for the detected hand.", "reading": ">=0.8 reliable.", "caveat": "Not a probability of being real."},
]
