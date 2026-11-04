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
    lib.ed_pairs_opt.argtypes = [_P_U8, _P_I64, _P_U8, _P_I64, ctypes.c_int64, _P_I32, ctypes.c_int]
    lib.ed_pairs_opt.restype = ctypes.c_int
    lib.ed_cross.argtypes = [_P_U8, _P_I64, ctypes.c_int64, _P_U8, _P_I64, ctypes.c_int64, _P_I32]
    lib.ed_cross.restype = ctypes.c_int
    lib.ed_cross_simd.argtypes = lib.ed_cross.argtypes
    lib.ed_cross_simd.restype = ctypes.c_int
    lib.ed_cross_simd_k.argtypes = lib.ed_cross.argtypes + [ctypes.c_int]
    lib.ed_cross_simd_k.restype = ctypes.c_int
    lib.ed_pairs_simd.argtypes = [_P_U8, _P_I64, _P_U8, _P_I64, ctypes.c_int64, _P_I32,
                                  ctypes.c_int, ctypes.c_int]
    lib.ed_pairs_simd.restype = ctypes.c_int
    lib.ed_simd_available.argtypes = []
    lib.ed_simd_available.restype = ctypes.c_int
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


def pairs(packed_a, packed_b, trim: bool = False) -> np.ndarray:
    (ba, oa), (bb, ob) = packed_a, packed_b
    n = len(oa) - 1
    out = np.empty(n, dtype=np.int32)
    if lib().ed_pairs_opt(ba, oa, bb, ob, n, out, int(trim)) != 0:
        raise MemoryError("ed_pairs allocation failed")
    return out


def cross(packed_q, packed_c) -> np.ndarray:
    (bq, oq), (bc, oc) = packed_q, packed_c
    nq, nc = len(oq) - 1, len(oc) - 1
    out = np.empty((nq, nc), dtype=np.int32)
    if lib().ed_cross(bq, oq, nq, bc, oc, nc, out) != 0:
        raise MemoryError("ed_cross allocation failed")
    return out


def simd_available() -> bool:
    return available() and bool(lib().ed_simd_available())


def cross_simd(packed_q, packed_c, k: int = 1) -> np.ndarray:
    """Phase 4: NEON inter-pair kernel (falls back to scalar without NEON).
    k = number of independent register groups interleaved (1, 2 or 4)."""
    (bq, oq), (bc, oc) = packed_q, packed_c
    nq, nc = len(oq) - 1, len(oc) - 1
    out = np.empty((nq, nc), dtype=np.int32)
    rc = lib().ed_cross_simd_k(bq, oq, nq, bc, oc, nc, out, int(k))
    if rc == -2:
        raise ValueError("k must be 1, 2 or 4")
    if rc != 0:
        raise MemoryError("ed_cross_simd allocation failed")
    return out


def pairs_simd(packed_a, packed_b, trim: bool = True, k: int = 4) -> np.ndarray:
    """Phase 4 step 2: NEON kernel for independent pairs (scalar without NEON)."""
    (ba, oa), (bb, ob) = packed_a, packed_b
    n = len(oa) - 1
    out = np.empty(n, dtype=np.int32)
    rc = lib().ed_pairs_simd(ba, oa, bb, ob, n, out, int(trim), int(k))
    if rc == -2:
        raise ValueError("k must be 1, 2 or 4")
    if rc != 0:
        raise MemoryError("ed_pairs_simd allocation failed")
    return out


def buildinfo() -> str:
    try:
        with open(BUILDINFO) as f:
            return f.read().strip()
    except OSError:
        return ""


# ---------------- CPython extension (reads str/bytes in place) ----------------

def _load_ext():
    import importlib.util
    import sysconfig
    path = os.path.join(_ROOT, "build", "_edpy" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("_edpy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXT = _load_ext()


def ext_available() -> bool:
    return _EXT is not None


def ext_pairs(A, B, trim: bool = True) -> np.ndarray:
    """A, B: sequences of str (all chars <= U+00FF) or bytes, same length."""
    out = np.empty(len(A), dtype=np.int32)
    _EXT.pairs(A, B, out, trim)
    return out


def ext_cross(Q, C) -> np.ndarray:
    out = np.empty((len(Q), len(C)), dtype=np.int32)
    _EXT.cross(Q, C, out)
    return out


def ext_pairs_simd(A, B, trim: bool = True, k: int = 4) -> np.ndarray:
    out = np.empty(len(A), dtype=np.int32)
    _EXT.pairs_simd(A, B, out, trim, k)
    return out
