"""A deterministic, dependency-free embedder for the bench.

Bag-of-words feature hashing: each word increments one of ``dim`` buckets, then
the vector is L2-normalised. It has NO real semantics (synonyms don't cluster),
but documents sharing words get correlated vectors — enough to exercise the
vector/hybrid retrieval paths reproducibly, with no model and no network.
"""

from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v
