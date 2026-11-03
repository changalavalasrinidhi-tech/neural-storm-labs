"""Tamper-evident chain of custody for the session log (torch-free).

Each entry stores the SHA-256 of the previous entry, so editing, deleting or re-ordering an entry in an
exported log breaks every hash after it. HONEST SCOPE: this proves the *exported file* is internally
consistent; it is not a digital signature and cannot prove who produced it or that the session
itself was not fabricated from the start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

GENESIS = "0" * 64
FIELDS = ("index", "time_utc", "kind", "source", "file_sha256", "summary", "prev_hash")


def _digest(entry: Dict[str, Any]) -> str:
    payload = json.dumps({k: entry.get(k) for k in FIELDS}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append(chain: List[Dict[str, Any]], kind: str, source: str, file_sha256: str, summary: str,
           time_utc: Optional[str] = None) -> Dict[str, Any]:
    entry = {"index": len(chain), "time_utc": time_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
             "kind": kind, "source": source, "file_sha256": file_sha256, "summary": summary,
             "prev_hash": chain[-1]["entry_hash"] if chain else GENESIS}
    entry["entry_hash"] = _digest(entry)
    chain.append(entry)
    return entry


def verify(chain: List[Dict[str, Any]]) -> Tuple[bool, Optional[int], str]:
    """Return (ok, first_bad_index, message)."""
    prev = GENESIS
    for i, e in enumerate(chain):
        if e.get("index") != i:
            return False, i, f"entry {i}: index out of order"
        if e.get("prev_hash") != prev:
            return False, i, f"entry {i}: previous-hash link is broken"
        if _digest(e) != e.get("entry_hash"):
            return False, i, f"entry {i}: contents do not match their hash"
        prev = e["entry_hash"]
    return True, None, f"{len(chain)} entr{'y' if len(chain) == 1 else 'ies'} verified" if chain else "chain is empty"


def head(chain: List[Dict[str, Any]]) -> str:
    return chain[-1]["entry_hash"] if chain else GENESIS


def export_json(chain: List[Dict[str, Any]]) -> str:
    ok, _, msg = verify(chain)
    return json.dumps({"format": "neural-storm-custody-v1", "head_hash": head(chain), "verified_at_export": ok, "message": msg,
                       "entries": chain}, indent=2)


def verify_export(text: str) -> Tuple[bool, str]:
    """Verify a previously exported custody JSON (for the 'check an exported log' box)."""
    try:
        doc = json.loads(text)
        chain = doc["entries"]
    except Exception as exc:
        return False, f"not a custody export: {exc}"
    ok, _, msg = verify(chain)
    if ok and doc.get("head_hash") != head(chain):
        return False, "head hash does not match the last entry"
    return ok, msg
