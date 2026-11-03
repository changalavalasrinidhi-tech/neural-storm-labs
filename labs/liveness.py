"""Experimental hand-gesture challenge-response ("liveness") check.

The app issues a random, time-limited instruction ("show 3 fingers"); the user answers with a
camera snapshot; the hand analytics verify it. A frame hash blocks re-submitting the same image.

HONEST SCOPE: this is a research prototype. A real-time face/hand deepfake or a pre-rendered
set of images can still pass a single-frame challenge. It raises the cost of replaying a *static*
file; it is not a security control and has not been validated against real attacks.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any, Dict, Iterable, Optional, Sequence

GESTURE_TARGETS = ("Open palm", "Fist", "Peace / victory", "Pointing", "Thumbs up")
DEFAULT_TTL_SECONDS = 90
MIN_SCORE = 0.70


def new_challenge(rng: Optional[secrets.SystemRandom] = None, now: Optional[float] = None,
                  ttl: int = DEFAULT_TTL_SECONDS) -> Dict[str, Any]:
    rng = rng or secrets.SystemRandom()
    now = time.time() if now is None else now
    if rng.random() < 0.5:
        n = rng.randint(1, 5)
        return {"id": secrets.token_hex(4), "kind": "fingers", "target": n, "issued_at": now, "ttl": ttl,
                "prompt": f"Hold up exactly {n} finger{'s' if n != 1 else ''} (thumb counts) and take a snapshot."}
    g = rng.choice(GESTURE_TARGETS)
    return {"id": secrets.token_hex(4), "kind": "gesture", "target": g, "issued_at": now, "ttl": ttl,
            "prompt": f"Show this gesture to the camera: {g}."}


def frame_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(challenge: Dict[str, Any], hands: Sequence[Dict[str, Any]], frame_sha: Optional[str] = None,
           used_hashes: Optional[Iterable[str]] = None, now: Optional[float] = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    age = now - float(challenge["issued_at"])
    base = {"challenge_id": challenge["id"], "age_seconds": round(age, 1)}
    if age > challenge["ttl"]:
        return {**base, "passed": False, "reason": f"Challenge expired after {challenge['ttl']} s - request a new one."}
    if frame_sha and used_hashes is not None and frame_sha in set(used_hashes):
        return {**base, "passed": False, "reason": "This exact image was already submitted (replay blocked)."}
    if not hands:
        return {**base, "passed": False, "reason": "No hand detected in the snapshot."}
    for h in hands:
        score = h.get("detector_score")
        if score is not None and score < MIN_SCORE:
            continue
        if challenge["kind"] == "fingers" and h["finger_count"] == challenge["target"]:
            return {**base, "passed": True, "reason": f"Detected {h['finger_count']} extended finger(s) as requested.", "observed": h["gesture"]}
        if challenge["kind"] == "gesture" and h["gesture"] == challenge["target"]:
            return {**base, "passed": True, "reason": f"Detected '{h['gesture']}' as requested.", "observed": h["gesture"]}
    seen = ", ".join(f"{h['gesture']} ({h['finger_count']} fingers)" for h in hands)
    return {**base, "passed": False, "reason": f"Did not match. Saw: {seen}.", "observed": seen}
