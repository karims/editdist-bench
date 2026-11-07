# editdist-bench

Fast exact edit distance (unit-cost Levenshtein, score only) for **batches of
short strings (≤ 256 characters)**, on Apple Silicon (NEON) and NVIDIA GPUs
(CUDA), measured under a protocol written before any measurement.

> Question: can bit-parallel edit distance (Myers 1999 / Hyyrö) be made
> meaningfully faster than the current state of the art for many short
> pairs, on one CPU core and on a GPU?

**Answer, in short:** yes, for short strings. On one Apple M4 core we are
2-10x faster than the best library at lengths 16-32. On an NVIDIA H100 our
CUDA search kernel is 2-3.5x faster than StringZilla's GPU backend, like for
like. The method is careful engineering of known algorithms (bit-parallel
Myers, SIMD lanes, latency hiding, exact prefix/suffix trimming), not a new
algorithm: time and space complexity are the same as Myers/Hyyrö.

## Results

All numbers below are the **lower of two separate runs**, every row passed a
correctness check against independent reference implementations, and the full
raw data is in `results/` (never edited). Pre-registered hypotheses and their
outcomes, including the misses: [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md).

Shapes: **search** = Q queries × C candidates (fuzzy search, `cdist`);
**pairs** = N independent pairs (`cpdist`). Grid: lengths 16-256 × error
rates 1-50% × alphabets (random bytes, DNA, English-like text).

### 1. One CPU core, Apple M4 (macOS, official runs)

Ours vs the best library on one core (RapidFuzz 3.14, StringZilla 5.3, edlib):

| Length | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| Search (kernel) | 8.2-9.6x | 4.0-5.0x | 1.4-1.8x | 0.9-1.0x | 0.7-0.8x |
| Pairs (end to end, from Python lists) | 2.1-2.9x | 1.9-2.5x | 1.8-2.2x | 1.4-1.9x | 1.2-1.4x |

- Search at L16: one M4 core (156 M pairs/s) beats StringZilla on all 10
  cores (114 M pairs/s).
- Above 64 characters StringZilla's single-thread search is faster (we have
  no NEON path there).
- Caveat: RapidFuzz has SIMD only on x86, not ARM; part of the lead on the
  Mac is that missing port.

On one x86 core (AMD EPYC 75F3), where RapidFuzz has its AVX2 code: our
scalar C still wins on pairs (L16: 31 vs 10 M pairs/s, mostly from trimming)
and on everything from L64 up, but RapidFuzz's AVX2 search is far ahead at
L16-32 (82 vs 19 M pairs/s). We have no x86 SIMD kernel.

### 2. GPU vs CPU on the same machine (RTX 4090 pod, 13 allowed CPU cores)

Kernel-only GPU throughput vs the best CPU result on that machine:

| Length | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| Search | 52-98x | 68-105x | 230-273x | 272-301x | 142-147x |
| Pairs | ~580x | ~460x | ~290x | ~300x | ~100x |

Counting the copy of inputs to the GPU and results back (from Python
lists):

- **Search** pays off from about 10,000-90,000 pairs per batch and reaches
  7.6x the 13-core CPU at 1M pairs.
- **Pairs** do not: packing a million Python strings and moving them over
  PCIe costs more than the GPU saves (1.2x the CPU at 1M pairs on an H100
  box, even with packing done in C). For independent pairs the GPU helps
  when the data already lives on the GPU or arrives packed.

### 3. GPU vs the GPU library (H100 SXM)

Ours vs StringZilla 5.3.2's CUDA backend (its Hopper path), both returning
results to Python, 2,000 × 2,000 search:

| Length | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| Ours / StringZilla | 2.0-2.5x | 2.4-2.6x | 1.8-3.0x | 2.5-3.4x | 2.8-3.1x |

- 17 of 20 buckets pass the claim rule; 3 are above the bar in both runs but
  within the noise of the shared host. None fails. The two runs were on the
  same pod ~15 minutes apart (weaker than separate days).
- StringZilla ran at its published speed (≈ 7.5 trillion cells/s at L256),
  and its correct outputs match ours on every batch.

### Issues found in StringZilla 5.3.2's CUDA backend

- **RTX 4090 (Ada, sm_89):** every call fails with
  `CUDA_ERROR_INVALID_VALUE`, even `"abc"` vs `"abd"` (it takes its `kepler`
  path there).
- **H100:** in one run, its `to_device`-inside-the-call variant returned 28
  wrong distances out of 300 checked on one batch (DNA, L64, 5% errors); the
  same input was correct in the other run. Caught by our correctness gate and
  excluded.

## What we ran

| Where | Hardware | Runs | Results |
|---|---|---|---|
| Mac (native) | Apple M4, 10 cores, macOS 26 | 2 full-grid official runs (+ 3 earlier quick-grid runs) | `results/macos-m4/` |
| RunPod | RTX 4090 + AMD EPYC 75F3 (13.6-CPU quota) | GPU / 13-core / 1-core full comparison, batch-size sweep | `results/gpu-NVIDIA-GeForce-RTX-4090/` |
| RunPod | H100 SXM + Intel Xeon 8480+ (23-CPU quota) | 2 head-to-head runs vs StringZilla GPU, pairs with C packing | `results/gpu-NVIDIA-H100-80GB-HBM3/` |
| Linux VM on the Mac | 4 vCPUs | development and preliminary numbers only | `results/vm-linux/` |

Every CSV has a matching `.env.json`: CPU, GPU, driver, OS, compiler flags,
library versions, git commit, machine load before/after, CPU quota.

## How it works

- **Core:** Myers' bit-parallel algorithm advances a whole DP column (up to
  64 cells) per text character with ~15 bit operations
  ([`csrc/myers.c`](csrc/myers.c), `block_step`, `run_text`).
- **Trimming:** exact, d(xPy, xQy) = d(P, Q); removes most work on
  near-identical pairs.
- **NEON (CPU):** 8/4/2 candidates per 128-bit register, and 4 independent
  register groups interleaved to hide the dependency chain
  (`DEFINE_KGROUP_KERNEL`, `DEFINE_PAIRS_KERNEL`).
- **CUDA:** pairs = one thread per pair, pattern in registers, Eq computed by
  4-byte SWAR compares; search = one block per query, its lookup table in
  shared memory ([`csrc/gpu_myers.cuh`](csrc/gpu_myers.cuh),
  [`csrc/gpu_kernels.cu`](csrc/gpu_kernels.cu)). The device functions also
  compile as plain C++ and are unit-tested on the CPU.
- **Python:** a C extension reads Python `str` (Latin-1) / `bytes` in place,
  and packs them in C for the GPU ([`csrc/edpy.c`](csrc/edpy.c)).

## How we built trust

- Hypotheses and pass/fail bars written before measuring
  ([`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)); misses are reported, and the
  one place we misquoted our own bar is corrected in the open.
- Correctness gate before every timing (300 outputs vs an independent
  reference), plus agreement checks across all contenders and across
  CPU/GPU runs (checksums).
- Measurement rules in [`docs/PROTOCOL.md`](docs/PROTOCOL.md): auto-calibrated
  samples (≥ 50 ms), interleaved shuffled timing, thread CPU time as a
  cross-check, machine state recorded, multi-core libraries limited to the
  CPUs the container is actually allowed, GPU / multi-core / single-core
  timed in separate processes, a claim needs two runs and must exceed
  1 + 2 × IQR.

## Reproduce

```bash
./setup.sh                                  # venv (Python >= 3.10), deps, C build, tests
source .venv/bin/activate
python -m bench.run_prior_art --quick       # a few minutes
scripts/official_run.sh --pairs 20000 --cross 200   # official macOS full grid (~35 min)
```

On a rented NVIDIA machine (image with the CUDA toolkit, e.g. RunPod PyTorch):

```bash
git clone https://github.com/karims/editdist-bench.git && cd editdist-bench
bash scripts/gpu_run.sh          # full GPU vs CPU comparison (~45 min)
bash scripts/gpu_run.sh h100     # head-to-head with StringZilla's GPU backend (~20 min)
```

Python ≥ 3.10 is required (StringZilla 5.x semantics). Results go to
`results/`; commit them as they are.

## Layout

```
docs/PROTOCOL.md       problem definition, buckets, metrics, measurement rules
docs/HYPOTHESES.md     pre-registered hypotheses and their outcomes
csrc/myers.c           CPU kernels: scalar Myers, trimming, NEON search and pairs
csrc/edpy.c            CPython extension: in-place str/bytes, packing
csrc/gpu_myers.cuh     GPU device functions (also compiled and tested on the CPU)
csrc/gpu_kernels.cu    CUDA kernels (compiled at run time by CuPy/NVRTC)
bench/datagen.py       deterministic data generator (lengths x error rates x alphabets)
bench/reference.py     ground truth: textbook DP + independent bit-parallel Myers
bench/baselines.py     every contender (libraries and ours) behind a correctness gate
bench/cmyers.py        Python bindings for the CPU kernels
bench/cudamyers.py     Python bindings for the GPU kernels
bench/run_prior_art.py benchmark runner -> results/*.csv + .env.json
bench/envinfo.py       environment capture; bench/sysload.py machine load, CPU quota
scripts/               official macOS run, GPU run, jitter probe
tests/                 ~200 tests (GPU ones skip without a GPU)
results/               raw benchmark output, by machine
```

## Limits

- Score only (no alignment/traceback), unit costs, byte alphabet.
- No early exit for a distance threshold ("≤ k?"), which many real users want.
- No SIMD path above 64 characters on the CPU, and no x86 SIMD kernel.
- Single-core CPU numbers; GPU numbers from rented, shared machines.
- Synthetic data (random bytes, DNA, English-like text with controlled edit
  rates).
