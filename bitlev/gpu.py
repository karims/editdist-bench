"""NVIDIA GPU versions of `pairwise` and `cdist` (needs `pip install bitlev[gpu]`,
i.e. CuPy, and an NVIDIA GPU with a CUDA 12+ driver). Kernels are compiled
on first use with NVRTC and cached by CuPy.

    from bitlev import gpu
    d = gpu.cdist(queries, choices, max_dist=2, dtype="uint8")

When it pays off (measured): search-style `cdist` from ~10^4-10^5 results per
call upward; `pairwise` mainly for long strings or when results stay on the
GPU (`return_device=True`), because moving many short strings costs more
than computing them on a CPU core.
"""
from __future__ import annotations

import os

import numpy as np

from . import _edpy
from ._common import as_list, check_dtype, check_k, convert

PAIR_CAPS = (16, 32, 64, 128, 256)   # ed_pairs_c<CAP>: shorter string <= CAP
CROSS_NWS = (1, 2, 4, 8)             # ed_cross_w<NW>: query <= 64 * NW
THREADS = 256
_mod = None


def _source_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "cuda"), os.path.join(os.path.dirname(here), "csrc")):
        if os.path.exists(os.path.join(d, "gpu_kernels.cu")):
            return d
    raise FileNotFoundError("CUDA sources not found (expected bitlev/cuda/gpu_kernels.cu)")


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
        d = _source_dir()
        src = open(os.path.join(d, "gpu_kernels.cu")).read()
        _mod = cp.RawModule(code=src, options=("-std=c++17", f"-I{d}"))
    return _mod


def _pack(strings):
    buf, off = _edpy.pack(strings)
    return np.frombuffer(buf, dtype=np.uint8), np.frombuffer(off, dtype=np.int64)


def _launch_dtype(dt):
    """Kernels write int32 or uint8; uint16 is produced from int32."""
    return dt == np.uint8


def _result(out, dt, return_device):
    import cupy as cp
    if dt == np.uint16:
        out = cp.minimum(out, 65535).astype(cp.uint16)
    return out if return_device else out.get()


def pairwise(a, b, max_dist=None, dtype=np.int32, return_device: bool = False):
    """out[i] = d(a[i], b[i]) on the GPU. Pairs whose shorter string is longer
    than 256 are computed on the CPU. return_device=True returns a CuPy array."""
    import cupy as cp
    a, b = as_list(a), as_list(b)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    k, dt = check_k(max_dist), check_dtype(dtype)
    n = len(a)
    out8 = _launch_dtype(dt)
    out = cp.empty(n, dtype=cp.uint8 if out8 else cp.int32)
    if n == 0:
        return _result(out, dt, return_device)
    pa, pb = _pack(a), _pack(b)
    m = np.minimum(np.diff(pa[1]), np.diff(pb[1]))
    cls = np.searchsorted(np.array(PAIR_CAPS), m)
    da = (cp.asarray(pa[0]), cp.asarray(pa[1]))
    db = (cp.asarray(pb[0]), cp.asarray(pb[1]))
    mod = _module()
    for ci, cap in enumerate(PAIR_CAPS):
        idx = np.nonzero(cls == ci)[0].astype(np.int32)
        if len(idx):
            mod.get_function(f"ed_pairs_c{cap}")(
                ((len(idx) + THREADS - 1) // THREADS,), (THREADS,),
                (da[0], da[1], db[0], db[1], cp.asarray(idx), np.int32(len(idx)), out,
                 np.int32(1), np.int32(k), np.int32(int(out8))))
    rest = np.nonzero(cls == len(PAIR_CAPS))[0]
    if len(rest):
        vals = np.empty(len(rest), dtype=np.int32)
        _edpy.pairs([a[i] for i in rest], [b[i] for i in rest], vals, True, k)
        out[cp.asarray(rest)] = cp.asarray(convert(vals, np.dtype(np.uint8)) if out8 else vals)
    cp.cuda.Device().synchronize()
    return _result(out, dt, return_device)


def cdist(queries, choices, max_dist=None, dtype=np.int32, return_device: bool = False):
    """out[i, j] = d(queries[i], choices[j]) on the GPU. Queries longer than
    512 are computed on the CPU. dtype=uint8 cuts the copy back by 4x."""
    import cupy as cp
    q, c = as_list(queries), as_list(choices)
    k, dt = check_k(max_dist), check_dtype(dtype)
    nq, nc = len(q), len(c)
    out8 = _launch_dtype(dt)
    out = cp.empty(nq * nc, dtype=cp.uint8 if out8 else cp.int32)
    if nq and nc:
        pq, pc = _pack(q), _pack(c)
        words = (np.diff(pq[1]) + 63) // 64
        cls = np.searchsorted(np.array(CROSS_NWS), words)
        dq = (cp.asarray(pq[0]), cp.asarray(pq[1]))
        dc = (cp.asarray(pc[0]), cp.asarray(pc[1]))
        gy = max(1, min((nc + THREADS - 1) // THREADS, 64))
        mod = _module()
        for ci, nw in enumerate(CROSS_NWS):
            idx = np.nonzero(cls == ci)[0].astype(np.int32)
            if len(idx):
                mod.get_function(f"ed_cross_w{nw}")(
                    (len(idx), gy), (THREADS,),
                    (dq[0], dq[1], cp.asarray(idx), dc[0], dc[1], np.int32(nc), out,
                     np.int32(k), np.int32(int(out8))),
                    shared_mem=256 * nw * 8)
        rest = np.nonzero(cls == len(CROSS_NWS))[0]
        if len(rest):
            vals = np.empty(len(rest) * nc, dtype=np.int32)
            _edpy.cross([q[i] for i in rest], c, vals, k)
            vals = convert(vals, np.dtype(np.uint8)) if out8 else vals
            out.reshape(nq, nc)[cp.asarray(rest)] = cp.asarray(vals.reshape(len(rest), nc))
        cp.cuda.Device().synchronize()
    return _result(out.reshape(nq, nc), dt, return_device)
