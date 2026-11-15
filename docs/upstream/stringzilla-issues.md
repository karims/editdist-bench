# Draft issues for StringZilla (github.com/ashvardanian/StringZilla)

Found while benchmarking StringZilla 5.3.2's CUDA backend against our own
kernels in https://github.com/karims/editdist-bench. Each issue below is
ready to paste. The repro script is `scripts/sz_gpu_repro.py` in that repo
(self-contained: only stringzilla, stringzillas-cuda and numpy).

---

## Issue 1 — `LevenshteinDistances` on GPU fails every call on RTX 4090 (`CUDA_ERROR_INVALID_VALUE`)

**Environment**
- stringzilla 5.3.2, stringzillas-cuda 5.3.2 (PyPI wheel), numpy 2.5.4, Python 3.12
- NVIDIA GeForce RTX 4090 (compute capability 8.9, 128 SMs, 24 GB)
- Driver 595.91.07 (CUDA 13.2); CUDA runtime bundled in the wheel 12.9
- Linux x86_64 (Ubuntu, kernel 6.8), RunPod container
- `stringzillas.__capabilities__` = `('serial', 'parallel', 'westmere', 'goldmont', 'haswell', 'cuda', 'kepler')`

**What happens**
Every GPU call raises, even for one-character-different strings:

```python
import numpy as np, stringzilla as sz, stringzillas as szs
gpu = szs.DeviceScope(gpu_device=0)
eng = szs.LevenshteinDistances(capabilities=gpu)
eng(szs.to_device(sz.Strs(["abc"])), szs.to_device(sz.Strs(["abd"])), device=gpu)
# ValueError: Levenshtein distances computation: CUDA_ERROR_INVALID_VALUE
```

Same result with or without `szs.to_device`, with empty strings, single
characters, lengths 64 / 300 / 1000, mixed lengths, and 3 × 5 batches. The
CPU backend works on the same machine. On an H100 (where the `hopper`
capability is listed) the same code works, so the failure seems specific to
the `kepler` path or to Ada GPUs.

**Expected**
`[[1]]`.

**Repro**: `python sz_gpu_repro.py basic`

---

## Issue 2 — GPU `LevenshteinDistances` intermittently returned wrong distances on H100

**Environment**
- stringzilla 5.3.2, stringzillas-cuda 5.3.2, numpy 2.5.4, Python 3.12.3
- NVIDIA H100 80GB HBM3 (SXM, compute capability 9.0, 132 SMs)
- Driver 595.91.07 (CUDA 13.2); bundled CUDA runtime 12.9
- Intel Xeon Platinum 8480+, Linux x86_64 (kernel 6.8), RunPod container
- capabilities include `cuda`, `kepler`, `hopper`

**What happened**
Our harness checks every contender before timing it: it computes a
15 × 20 matrix (300 distances) and compares each value with an independent
reference implementation. In one of two otherwise identical runs, ~15
minutes apart on the same machine, the call below returned **28 wrong
distances out of 300**:

```python
gpu = szs.DeviceScope(gpu_device=0)
eng = szs.LevenshteinDistances(capabilities=gpu)
eng(szs.to_device(sz.Strs(Q)), szs.to_device(sz.Strs(C)), device=gpu)   # Q: 15, C: 20 strings
```

Data: DNA (ACGT), base length 64, ~3 random edits per string (5% edit rate),
queries and candidates drawn from a few shared base strings. The same input
was correct in the other run, and the variant that calls `to_device` once
before timing (instead of inside each call) was correct on it in the same
run. All other 20 + 19 batches agreed exactly with our reference.

So: intermittent, seen once, on the path that moves inputs with
`to_device` inside the call. It may be a race between the unified-memory
copy and the kernel launch, but that is a guess.

**Repro attempt**: `python sz_gpu_repro.py stress` runs 2,000 calls of that
shape and compares GPU against StringZilla's own CPU backend. We have not
yet run it on an H100, so we cannot say how often it triggers.

**Expected**
GPU results identical to the CPU backend's.

---

Happy to run more diagnostics on either if useful.
