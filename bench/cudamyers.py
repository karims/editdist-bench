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
        self._rest_cache = {}
        if len(self.rest):
            self.rest_a = _subset(packed_a, self.rest)
            self.rest_b = _subset(packed_b, self.rest)
            self.rest_idx = cp.asarray(self.rest.astype(np.int64))

    def rest_values(self, k: int):
        """Pairs longer than 256: computed once on the CPU (per threshold)."""
        if k not in self._rest_cache:
            import cupy as cp
            from bench import cmyers
            vals = (cmyers.pairs(self.rest_a, self.rest_b, trim=True) if k < 0
                    else cmyers.pairs_k(self.rest_a, self.rest_b, k))
            self._rest_cache[k] = cp.asarray(vals)
        return self._rest_cache[k]


def _out(n: int, out8: bool):
    import cupy as cp
    return cp.empty(n, dtype=cp.uint8 if out8 else cp.int32)


def _finish(out, idx, vals, out8):
    if out8:
        import cupy as cp
        vals = cp.minimum(vals, 255).astype(cp.uint8)
    out[idx] = vals


def pairs(st: PairsState, trim: bool = True, k: int = -1, out8: bool = False):
    """Distances on the GPU; returns a CuPy array (still on the GPU): int32,
    or uint8 saturating at 255 with out8. k >= 0: min(distance, k + 1)."""
    mod = _module()
    out = _out(st.n, out8)
    for cap, idx, cnt in st.groups:
        mod.get_function(f"ed_pairs_c{cap}")(
            ((cnt + THREADS - 1) // THREADS,), (THREADS,),
            (st.a[0], st.a[1], st.b[0], st.b[1], idx, np.int32(cnt), out,
             np.int32(int(trim)), np.int32(k), np.int32(int(out8))))
    if len(st.rest):
        _finish(out, st.rest_idx, st.rest_values(k), out8)
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


def cross(st: CrossState, k: int = -1, out8: bool = False):
    """nq x nc distances (row-major, flat) as a CuPy array on the GPU: int32,
    or uint8 saturating at 255 with out8. k >= 0: min(distance, k + 1)."""
    import cupy as cp
    mod = _module()
    out = _out(st.nq * st.nc, out8)
    # enough candidate-blocks per query to cover nc, capped so the grid stays sane
    gy = max(1, min((st.nc + THREADS - 1) // THREADS, 64))
    for nw, idx, cnt in st.groups:
        mod.get_function(f"ed_cross_w{nw}")(
            (cnt, gy), (THREADS,),
            (st.q[0], st.q[1], idx, st.c[0], st.c[1], np.int32(st.nc), out,
             np.int32(k), np.int32(int(out8))),
            shared_mem=256 * nw * 8)
    if len(st.rest):
        vals = st.rest_vals if k < 0 else cp.minimum(st.rest_vals, k + 1)
        if out8:
            vals = cp.minimum(vals, 255).astype(cp.uint8)
        out.reshape(st.nq, st.nc)[st.rest_idx] = vals
    sync()
    return out
