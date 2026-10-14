"""Byte integrity hashes and domain-separated host identities."""

from __future__ import annotations

from hashlib import sha256


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()
