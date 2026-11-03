"""ctypes wrapper for the plain C Myers kernels (csrc/myers.c, built by `make`)."""

from __future__ import annotations

import ctypes
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(_ROOT, "build", "libmyers.so")
BUILDINFO = os.path.join(_ROOT, "build", "BUILDINFO")

_P_U8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_P_I64 = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
_P_I32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")


def available() -> bool:
    return os.path.exists(LIB_PATH)


def _lib():
    if not available():
        raise ImportError(f"{LIB_PATH} not found; run `make` in the repo root")
    lib = ctypes.CDLL(LIB_PATH)
    lib.ed_pairs.argtypes = [_P_U8, _P_I64, _P_U8, _P_I64, ctypes.c_int64, _P_I32]
    lib.ed_pairs.restype = ctypes.c_int
    lib.ed_cross.argtypes = [_P_U8, _P_I64, ctypes.c_int64, _P_U8, _P_I64, ctypes.c_int64, _P_I32]
    lib.ed_cross.restype = ctypes.c_int
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _lib()
    return _LIB


def pack(strings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate strings into one byte buffer plus offsets (1 char = 1 byte)."""
    enc = [s.encode("latin-1") for s in strings]
    off = np.zeros(len(enc) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in enc], out=off[1:])
    buf = np.frombuffer(b"".join(enc) or b"\0", dtype=np.uint8).copy()
    return buf, off


def pairs(packed_a, packed_b) -> np.ndarray:
    (ba, oa), (bb, ob) = packed_a, packed_b
    n = len(oa) - 1
    out = np.empty(n, dtype=np.int32)
    if lib().ed_pairs(ba, oa, bb, ob, n, out) != 0:
        raise MemoryError("ed_pairs allocation failed")
    return out


def cross(packed_q, packed_c) -> np.ndarray:
    (bq, oq), (bc, oc) = packed_q, packed_c
    nq, nc = len(oq) - 1, len(oc) - 1
    out = np.empty((nq, nc), dtype=np.int32)
    if lib().ed_cross(bq, oq, nq, bc, oc, nc, out) != 0:
        raise MemoryError("ed_cross allocation failed")
    return out


def buildinfo() -> str:
    try:
        with open(BUILDINFO) as f:
            return f.read().strip()
    except OSError:
        return ""
