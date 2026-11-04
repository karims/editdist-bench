"""CUDA kernels (csrc/gpu_kernels.cu) through CuPy, which compiles them with
NVRTC on first use. Needs an NVIDIA GPU and `pip install cupy-cuda12x`.

Host-side bookkeeping (which pairs/queries go to which kernel) is plain numpy
and tested without a GPU (tests/test_cuda_host.py); the GPU-only parts are the
kernel launches and copies.
"""
from __future__ import annotations

import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSRC = os.path.join(_ROOT, "csrc")
PAIR_CAPS = (16, 32, 64, 128, 256)          # kernel ed_pairs_c<CAP>
CROSS_NWS = (1, 2, 4, 8)                    # kernel ed_cross_w<NW>, pattern <= 64*NW
THREADS = 256
_mod = None


# ---------------------------------------------------------------- host side --
def pair_classes(off_a: np.ndarray, off_b: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Indices per pair kernel (by the shorter length) and the rest (> 256, CPU)."""
    m = np.minimum(np.diff(off_a), np.diff(off_b))
    cls = np.searchsorted(np.array(PAIR_CAPS), m)        # m <= CAP -> that class
    groups = [np.nonzero(cls == k)[0].astype(np.int32) for k in range(len(PAIR_CAPS))]
    return groups, np.nonzero(cls == len(PAIR_CAPS))[0]


def query_classes(off_q: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Query indices per cross kernel (by query length) and the rest (> 512, CPU)."""
    words = (np.diff(off_q) + 63) // 64                  # 0 for an empty query -> NW 1
    cls = np.searchsorted(np.array(CROSS_NWS), words)
    groups = [np.nonzero(cls == k)[0].astype(np.int32) for k in range(len(CROSS_NWS))]
    return groups, np.nonzero(cls == len(CROSS_NWS))[0]


def _subset(packed, idx):
    """Host (buf, off) for a subset of strings, for the CPU fallback."""
    buf, off = packed
    parts = [buf[off[i]:off[i + 1]] for i in idx]
    o = np.zeros(len(parts) + 1, dtype=np.int64)
    o[1:] = np.cumsum([len(p) for p in parts])
    return (np.concatenate(parts) if parts else np.zeros(1, np.uint8)), o


# ----------------------------------------------------------------- GPU side --
def available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _module():
    global _mod
    if _mod is None:
        import cupy as cp
        src = open(os.path.join(_CSRC, "gpu_kernels.cu")).read()
        _mod = cp.RawModule(code=src, options=("-std=c++17", f"-I{_CSRC}"))
    return _mod


def device_info() -> dict:
    import cupy as cp
    p = cp.cuda.runtime.getDeviceProperties(0)
    return {"gpu": p["name"].decode() if isinstance(p["name"], bytes) else p["name"],
            "compute_capability": f'{p["major"]}.{p["minor"]}',
            "sm_count": p["multiProcessorCount"],
            "memory_gb": round(p["totalGlobalMem"] / 2**30, 1),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "cupy": cp.__version__}


def sync():
    import cupy as cp
    cp.cuda.Device().synchronize()


class PairsState:
    """Packed pairs resident on the GPU, plus the class split."""

    def __init__(self, packed_a, packed_b):
        import cupy as cp
        self.host_a, self.host_b = packed_a, packed_b
        self.n = len(packed_a[1]) - 1
        self.a = tuple(cp.asarray(x) for x in packed_a)
        self.b = tuple(cp.asarray(x) for x in packed_b)
        groups, self.rest = pair_classes(packed_a[1], packed_b[1])
        self.groups = [(cap, cp.asarray(g), len(g)) for cap, g in zip(PAIR_CAPS, groups) if len(g)]
        if len(self.rest):
            from bench import cmyers
            self.rest_vals = cp.asarray(cmyers.pairs(_subset(packed_a, self.rest),
                                                     _subset(packed_b, self.rest), trim=True))
            self.rest_idx = cp.asarray(self.rest.astype(np.int64))


def pairs(st: PairsState, trim: bool = True):
    """Distances on the GPU; returns a CuPy int32 array (still on the GPU)."""
    import cupy as cp
    mod = _module()
    out = cp.empty(st.n, dtype=cp.int32)
    for cap, idx, cnt in st.groups:
        mod.get_function(f"ed_pairs_c{cap}")(
            ((cnt + THREADS - 1) // THREADS,), (THREADS,),
            (st.a[0], st.a[1], st.b[0], st.b[1], idx, np.int32(cnt), out, np.int32(int(trim))))
    if len(st.rest):
        out[st.rest_idx] = st.rest_vals          # > 256: computed once on the CPU
    sync()
    return out


class CrossState:
    def __init__(self, packed_q, packed_c):
        import cupy as cp
        self.nq, self.nc = len(packed_q[1]) - 1, len(packed_c[1]) - 1
        self.q = tuple(cp.asarray(x) for x in packed_q)
        self.c = tuple(cp.asarray(x) for x in packed_c)
        groups, self.rest = query_classes(packed_q[1])
        self.groups = [(nw, cp.asarray(g), len(g)) for nw, g in zip(CROSS_NWS, groups) if len(g)]
        if len(self.rest):
            from bench import cmyers
            sub = cmyers.cross(_subset(packed_q, self.rest), packed_c).reshape(len(self.rest), -1)
            self.rest_vals = cp.asarray(sub)
            self.rest_idx = cp.asarray(self.rest.astype(np.int64))


def cross(st: CrossState):
    """nq x nc distances (row-major, flat) as a CuPy int32 array on the GPU."""
    import cupy as cp
    mod = _module()
    out = cp.empty(st.nq * st.nc, dtype=cp.int32)
    # enough candidate-blocks per query to cover nc, capped so the grid stays sane
    gy = max(1, min((st.nc + THREADS - 1) // THREADS, 64))
    for nw, idx, cnt in st.groups:
        mod.get_function(f"ed_cross_w{nw}")(
            (cnt, gy), (THREADS,),
            (st.q[0], st.q[1], idx, st.c[0], st.c[1], np.int32(st.nc), out),
            shared_mem=256 * nw * 8)
    if len(st.rest):
        out.reshape(st.nq, st.nc)[st.rest_idx] = st.rest_vals
    sync()
    return out
