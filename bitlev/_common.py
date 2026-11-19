"""Shared input checks and output conversion."""
from __future__ import annotations

import numpy as np

DTYPES = (np.int32, np.uint16, np.uint8)


def check_k(max_dist) -> int:
    if max_dist is None:
        return -1
    k = int(max_dist)
    if k < 0:
        raise ValueError("max_dist must be >= 0 (or None)")
    return min(k, 2**30)


def check_dtype(dtype):
    dt = np.dtype(dtype)
    if dt not in [np.dtype(d) for d in DTYPES]:
        raise ValueError("dtype must be numpy int32, uint16 or uint8")
    return dt


def convert(out: np.ndarray, dt) -> np.ndarray:
    """int32 -> requested dtype; narrower types saturate at their maximum."""
    if dt == np.int32:
        return out
    return np.minimum(out, np.iinfo(dt).max).astype(dt)


def as_list(x):
    return x if isinstance(x, list) else list(x)
