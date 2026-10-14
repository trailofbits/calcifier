"""Pinned standard vectors and explicit composite-identity boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from calcifier.hashing import sha256_hex

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "sequencehash-sha256.json").read_text(encoding="utf-8")
)["vectors"]


def test_raw_integrity_hash_keeps_standard_sha256_encoding():
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
