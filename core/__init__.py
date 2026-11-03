"""Neural Storm FT-02 core detection package.

Exposes the ingestion router, dual-domain visual detector, audio spoof
detector, lip-sync engine, calibrated fusion layer, and forensic explainer.
"""

from core.ingestion import IngestedMedia, MediaIngestor
from core.visual_detector import DualDomainVisualDetector
from core.audio_detector import AudioSpoofDetector
from core.sync_engine import LipSyncEngine
from core.fusion import CalibratedEvidenceFusion, FusionResult
from core.explainability import ForensicExplainer, ExplanationBundle

__all__ = [
    "IngestedMedia",
    "MediaIngestor",
    "DualDomainVisualDetector",
    "AudioSpoofDetector",
    "LipSyncEngine",
    "CalibratedEvidenceFusion",
    "FusionResult",
    "ForensicExplainer",
    "ExplanationBundle",
]

__version__ = "1.0.0"
