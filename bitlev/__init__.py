"""bitlev: fast exact edit distance (unit-cost Levenshtein) for batches of
short strings.

    import bitlev
    bitlev.distance("kitten", "sitting")                  # 3
    bitlev.pairwise(["abc", "kitten"], ["abd", "sitting"]) # array([1, 3])
    bitlev.cdist(queries, choices)                         # len(queries) x len(choices)
    bitlev.cdist(queries, choices, max_dist=2, dtype="uint8")

Strings are compared byte by byte: pass `str` whose characters are all
<= U+00FF (ASCII, Latin-1) or `bytes`. For other text, encode first (e.g.
UTF-8); the distance is then counted in bytes.

`max_dist=k` returns min(distance, k + 1) ("more than k" is k + 1). For
longer strings, and whenever lengths differ by more than k, hopeless pairs
stop early, which can be several times faster; for short strings the result
is simply capped. `dtype` may be int32 (default),
uint16 or uint8; narrower types saturate at their maximum.

GPU versions (NVIDIA, needs `pip install bitlev[gpu]`): `bitlev.gpu`.
"""
from __future__ import annotations

import numpy as np

from . import _edpy
from ._common import as_list, check_dtype, check_k, convert

__version__ = "0.1.0"
__all__ = ["distance", "pairwise", "cdist", "simd_available"]


def simd_available() -> bool:
    """True when the ARM NEON kernels are compiled in (Apple Silicon, arm64)."""
    return _SIMD


_SIMD = bool(_edpy.simd_available())


def distance(a, b, max_dist=None) -> int:
    """Edit distance of one pair (min(d, max_dist + 1) with max_dist)."""
    out = np.empty(1, dtype=np.int32)
    _edpy.pairs([a], [b], out, True, check_k(max_dist))
    return int(out[0])


def _avg_len(xs) -> float:
    s = xs[:256]
    return sum(map(len, s)) / max(1, len(s))


def pairwise(a, b, max_dist=None, dtype=np.int32, method: str = "auto") -> np.ndarray:
    """Distances of independent pairs: out[i] = d(a[i], b[i]).

    method: "auto", "scalar" (exact prefix/suffix trimming, one pair at a
    time) or "simd" (NEON, several pairs per register; ARM only)."""
    a, b = as_list(a), as_list(b)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    k, dt = check_k(max_dist), check_dtype(dtype)
    out = np.empty(len(a), dtype=np.int32)
    if method == "auto":
        # NEON pairs won only around 33-64 characters in our measurements;
        # with a threshold, the scalar path's early exit wins.
        method = "simd" if (_SIMD and k < 0 and 32 < _avg_len(a) <= 64) else "scalar"
    if method == "simd":
        if not _SIMD:
            raise ValueError("method='simd' needs ARM NEON")
        _edpy.pairs_simd(a, b, out, True, 4)
        if k >= 0:
            np.minimum(out, k + 1, out=out)
    elif method == "scalar":
        _edpy.pairs(a, b, out, True, k)
    else:
        raise ValueError("method must be 'auto', 'scalar' or 'simd'")
    return convert(out, dt)


def cdist(queries, choices, max_dist=None, dtype=np.int32, method: str = "auto") -> np.ndarray:
    """All-pairs distances: out[i, j] = d(queries[i], choices[j]).

    method: "auto", "scalar" (with early exit when max_dist is set) or
    "simd" (NEON, 8/4/2 choices per register for lengths <= 64)."""
    q, c = as_list(queries), as_list(choices)
    k, dt = check_k(max_dist), check_dtype(dtype)
    out = np.empty(len(q) * len(c), dtype=np.int32)
    if method == "auto":
        # Measured on Apple M4 (150 x 1,500, DNA, random and near-duplicate):
        # NEON wins up to ~48 characters even with a threshold (2-8x); above
        # that, a threshold's early exit wins (up to 6x at 128), and without a
        # threshold NEON still wins up to 64.
        L = _avg_len(q)
        method = "simd" if _SIMD and (L <= 48 or (k < 0 and L <= 64)) else "scalar"
    if method == "simd":
        _edpy.cross_simd(q, c, out, 4, k)
    elif method == "scalar":
        _edpy.cross(q, c, out, k)
    else:
        raise ValueError("method must be 'auto', 'scalar' or 'simd'")
    return convert(out.reshape(len(q), len(c)), dt)
