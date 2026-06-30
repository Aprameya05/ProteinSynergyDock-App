"""
audit_log.py

Append-only, hash-chained audit log for every synergy prediction made
through the app or API.

Why hash-chained: a plain append-only log can still be silently edited
after the fact (rows deleted/altered) without detection. Chaining each
entry's hash into the next entry means any tampering breaks the chain
and is detectable by verify_chain(). This is the same pattern used in
HIPAA-adjacent audit requirements and matches the hash-chained record
design already used in the IoT sterilization patent, so it's a
deliberate, recognizable pattern rather than a one-off.

Storage: JSONL file, one entry per line, written via SQLite-free plain
file I/O so it has zero extra dependencies and works unmodified on
Streamlit Cloud's ephemeral filesystem (note: Streamlit Cloud's
filesystem resets on redeploy — for production use this would point at
persistent storage; documented honestly rather than glossed over).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional


GENESIS_HASH = "0" * 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash_entry(entry: dict) -> str:
    """SHA-256 of the entry's canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _input_hash(drug_a: str, drug_b: str, cell_line: str) -> str:
    """Hash of the request inputs only — lets you detect duplicate/replayed
    requests without storing PII-shaped data twice."""
    canonical = json.dumps(
        {"drug_a": drug_a, "drug_b": drug_b, "cell_line": cell_line},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class AuditLog:
    def __init__(self, path: str = "audit_log.jsonl"):
        self.path = path

    def _last_hash(self) -> str:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return GENESIS_HASH
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            buf = b""
            # read backwards to find the last line efficiently
            while pos > 0:
                pos -= 1
                f.seek(pos)
                char = f.read(1)
                if char == b"\n" and buf:
                    break
                buf = char + buf
            last_line = buf.decode("utf-8").strip()
        if not last_line:
            return GENESIS_HASH
        last_entry = json.loads(last_line)
        return last_entry["entry_hash"]

    def record(
        self,
        drug_a: str,
        drug_b: str,
        cell_line: str,
        output_resource_type: str,
        output_summary: str,
        model_version: str,
        success: bool,
        user: Optional[str] = "anonymous",
    ) -> dict:
        """
        Appends one audit entry and returns it. Call this on every
        prediction request, success or failure — failed/invalid requests
        are audited too, since "who tried what and when" matters even
        when nothing was produced.
        """
        prev_hash = self._last_hash()
        entry = {
            "timestamp": _now_iso(),
            "user": user,
            "input_hash": _input_hash(drug_a, drug_b, cell_line),
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "model_version": model_version,
            "output_resource_type": output_resource_type,  # "DiagnosticReport" | "OperationOutcome"
            "output_summary": output_summary,
            "success": success,
            "prev_hash": prev_hash,
        }
        entry["entry_hash"] = _hash_entry(entry)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        return entry

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """
        Verifies hash-chain integrity across the entire log.
        Returns (is_valid, first_broken_index_or_None).
        """
        entries = self.read_all()
        expected_prev = GENESIS_HASH
        for i, entry in enumerate(entries):
            if entry["prev_hash"] != expected_prev:
                return False, i
            claimed_hash = entry["entry_hash"]
            recomputed = dict(entry)
            del recomputed["entry_hash"]
            if _hash_entry(recomputed) != claimed_hash:
                return False, i
            expected_prev = claimed_hash
        return True, None
