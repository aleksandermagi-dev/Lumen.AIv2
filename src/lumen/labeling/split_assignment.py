from __future__ import annotations

import hashlib


def stable_split_assignment(*parts: str) -> str:
    stable_key = "|".join(str(part or "") for part in parts)
    bucket = int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
