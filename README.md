# editdist-bench

Fast exact edit distance (unit-cost Levenshtein, score only) for **batches of
short strings (≤ 256 characters)**, on Apple Silicon (NEON) and NVIDIA GPUs
(CUDA), measured under a protocol written before any measurement.

> Question: can bit-parallel edit distance (Myers 1999 / Hyyrö) be made
> meaningfully faster than the current state of the art for many short
> pairs, on one CPU core and on a GPU?

**Answer, in short:** yes, for short strings. On one Apple M4 core we are
2-9x faster than the best library at lengths 16-32 (e.g. search at L16:
160 vs 17 M pairs/s). On an NVIDIA H100 our CUDA search kernel is 2-3.5x
faster than StringZilla's GPU backend, like for like (e.g. L64: 1,500 vs
500 M pairs/s). The method is careful engineering of known algorithms (bit-parallel
Myers, SIMD lanes, latency hiding, exact prefix/suffix trimming), not a new
algorithm: time and space complexity are the same as Myers/Hyyrö.

## Results

**How to read the tables:** numbers are throughput in **millions of
distance computations per second** (M pairs/s); higher is better. Each table
shows one representative condition (random bytes, 5% edits) at every length;
the other alphabets (DNA, English-like text) and error rates (1-50%) look the
same and are in the raw CSVs. "Speedup" = ours ÷ the best library in that
column. Every row passed a correctness check against independent reference
implementations.

Two shapes are measured:
- **Search**: Q queries × C candidates, every query against every candidate
  (fuzzy search; RapidFuzz `cdist`).
- **Pairs**: N independent pairs (RapidFuzz `cpdist`), timed **end to end**
  from Python lists of strings.

### 1. One CPU core: Apple M4 (macOS)

Official run of 2026-11-05; the run of 2026-11-04 agrees (median difference
0.3%). Search: 200 × 200; pairs: 20,000.

**Search**

| One core, M pairs/s | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| RapidFuzz 3.14 (`cdist`) | 17.1 | 8.2 | 2.7 | 0.78 | 0.28 |
| StringZilla 5.3 (`LevenshteinDistances`) | 17.0 | 10.5 | 4.0 | 1.30 | 0.48 |
| edlib 1.3.9 (one call per pair) | 0.46 | 0.28 | 0.17 | 0.09 | 0.05 |
| **Ours** (NEON, 4 register groups) | **159.6** | **41.4** | **5.5** | 1.12 | 0.34 |
| **Speedup vs best library** | **9.4x** | **4.0x** | **1.4x** | 0.86x | 0.71x |
| *For scale: StringZilla on all 10 cores* | *139.6* | *79.6* | *32.7* | *11.5* | *4.2* |

**Pairs (end to end from Python lists)**

| One core, M pairs/s | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| RapidFuzz 3.14 (`cpdist`) | 14.0 | 6.3 | 3.7 | 1.00 | 0.31 |
| StringZilla 5.3 (one call per pair¹) | 0.70 | 0.60 | 0.48 | 0.38 | 0.22 |
| edlib 1.3.9 (one call per pair) | 0.62 | 0.38 | 0.22 | 0.10 | 0.05 |
| **Ours**, scalar + trimming | **41.0** | 12.0 | 4.9 | **1.48** | **0.39** |
| **Ours**, NEON pairs | 38.8 | **13.1** | **6.9** | 1.47 | 0.38 |
| **Speedup vs best library** | **2.9x** | **2.1x** | **1.9x** | **1.5x** | **1.3x** |
| *For scale: RapidFuzz on all 10 cores* | *9.9* | *5.6* | *3.7* | *1.4* | *0.8* |

¹ StringZilla 5.x has no pairwise API (only query × candidate matrices), so
each pair is a 1 × 1 matrix call; this row mostly measures call overhead.

Notes:
- At L16 our single core (160 M/s) is faster than StringZilla on all 10
  cores (140 M/s). Above 64 characters StringZilla's single-thread search
  wins: we have no NEON path there.
- RapidFuzz has SIMD only on x86, not ARM; part of our lead on the Mac is
  that missing port. On an x86 core (table 2) RapidFuzz's AVX2 search beats
  ours at L16-32.

### 2. GPU vs CPU on the same machine: RTX 4090 + AMD EPYC 75F3

RunPod; the container is allowed 13 CPU cores. Search: 2,000 × 2,000;
pairs: 1,000,000 (single-core rows: 300 × 300 and 100,000; throughput does
not depend on batch size at these sizes).

**Search**

| M pairs/s | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| RapidFuzz, 1 core (AVX2) | 82.4 | 35.2 | 3.2 | 0.7 | 0.2 |
| StringZilla, 1 core | 17.4 | 9.7 | 2.9 | 0.6 | 0.2 |
| Ours, 1 core (scalar C; no x86 SIMD) | 18.9 | 9.6 | 3.7 | 1.1 | 0.3 |
| RapidFuzz, 13 cores | 402.7 | 201.6 | 13.1 | 3.4 | 1.2 |
| StringZilla, 13 cores | 138.3 | 83.1 | 31.3 | 7.8 | 2.3 |
| **Ours, GPU, compute only** (data on the GPU) | **35,171** | **21,120** | **7,805** | **2,111** | **336** |
| **Ours, GPU, incl. copying** from Python lists | **1,712** | **1,759** | **1,337** | **875** | **273** |
| Speedup, GPU compute only vs best CPU | 87x | 105x | 249x | 271x | 146x |
| Speedup, GPU incl. copying vs best CPU | 4.3x | 8.7x | 43x | 112x | 119x |

**Pairs**

| M pairs/s | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| RapidFuzz, 1 core | 10.3 | 5.3 | 3.5 | 0.9 | 0.3 |
| Ours, 1 core (scalar + trimming) | 30.9 | 11.1 | 5.2 | 1.4 | 0.4 |
| RapidFuzz, 13 cores | 7.3 | 7.9 | 7.1 | 3.4 | 1.1 |
| **Ours, GPU, compute only** | **22,223** | **5,380** | **1,583** | **451** | **120** |
| Ours, GPU, incl. copying (from packed arrays) | 79.5 | 67.0 | 58.4 | 41.3 | 22.5 |
| Ours, GPU, end to end from Python lists² | 13.1 | 10.4 | 7.6 | 4.6 | 2.5 |
| Speedup, GPU compute only vs best CPU | 719x | 485x | 223x | 133x | 109x |
| Speedup, GPU end to end vs best CPU | 0.4x | 0.9x | 1.1x | 1.4x | 2.3x |

² Measured on the H100 machine (table 3) after moving string packing into C;
on this machine packing was still in Python (2.4 M/s at L16).

What this means:
- The GPU computes 100-700x faster than the best CPU result.
- For **search** that survives the copying: 4-119x end to end. The GPU wins
  from about 10,000-90,000 pairs per batch (smaller batches: CPU wins).
- For **pairs** it does not: packing a million Python strings and moving
  them to the GPU costs more than the computation. End to end, one CPU core
  running our C code is about as fast as the GPU up to L64. The GPU pays off
  for pairs when the data is already on the GPU or arrives packed.
- Short-pair compute-only rates are flattered by the 4090's 72 MB cache
  holding the repeated batch.

### 3. GPU vs the GPU library: H100 SXM

Ours vs StringZilla 5.3.2's CUDA backend (its H100 "Hopper" path). Both
return results to Python, so this is like for like. Search 2,000 × 2,000;
two runs on the same pod, ~15 minutes apart.

| M pairs/s, run 1 / run 2 | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| StringZilla GPU (better of its 2 variants) | 250 / 640 | 750 / 730 | 470 / 500 | 260 / 260 | 120 / 120 |
| **Ours, GPU** | **510 / 1,920** | **1,880 / 2,030** | **1,480 / 1,500** | **840 / 890** | **330 / 340** |
| **Speedup** | **2.0x / 3.0x** | **2.5x / 2.8x** | **3.1x / 3.0x** | **3.2x / 3.4x** | **2.8x / 2.9x** |
| *Ours, compute only (not like for like)* | *27,150* | *15,990* | *5,970* | *1,720* | *430* |

- Across all 20 tested conditions (2 alphabets × 2 error rates × 5 lengths)
  ours is 1.8-4.6x faster in both runs; 17 pass our claim rule, 3 are above
  the bar but within the noise of the shared machine, none fails.
- StringZilla ran at its published speed (about 7.5 trillion DP cells/s at
  L256), and its correct outputs match ours on every batch.
- The L16 column is noisy in run 1 (spread up to 70%) because of the shared
  host's CPU, which handles the copies.

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
