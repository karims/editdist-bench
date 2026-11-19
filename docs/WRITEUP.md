# Faster edit distance for short strings, measured honestly

*A small algorithm-engineering project: bit-parallel Levenshtein distance for
batches of short strings on Apple Silicon and NVIDIA GPUs, with every
hypothesis written down before measuring. Code, raw data and the library:
[github.com/karims/editdist-bench](https://github.com/karims/editdist-bench).*

## The question

Edit distance (how many single-character insertions, deletions or
substitutions turn one string into another) is everywhere: fuzzy search,
deduplicating records, matching DNA barcodes, checking CRISPR guides against
a genome. Most of these workloads compute it **millions of times on short
strings**: 16 to 256 characters.

The best general approach has been known since 1999. Gene Myers showed how to
store a whole column of the dynamic-programming table as two bitmasks and
advance it with about fifteen bit operations, 64 cells at a time. Mature
libraries (RapidFuzz, StringZilla, edlib) build on it. So the question was
narrow: **can careful engineering still beat them, by a margin worth
reporting, on one CPU core and on a GPU?**

## How we kept it honest

Before writing any fast code we wrote down the problem, the workload grid
(lengths 16-256, error rates 1-50%, random bytes / DNA / English-like text),
the measurement rules, and the hypotheses with their pass/fail bars, for
example "≥ 1.5x the best library at lengths ≤ 64". Then:

- Every contender, ours and theirs, passes a correctness check against
  independent reference implementations before it is timed.
- A claim needs **two separate runs**, and the speedup must exceed the
  measured noise.
- Raw results are committed and never edited, including the misses.

That protocol caught real problems. A Linux VM made one of our kernels look
slower than it is on the actual chip. A rented GPU machine let libraries
start 128 threads on a 13.6-CPU allowance, throttling them to a fraction of
single-core speed; we limited them to the CPUs actually allowed and re-ran.
It also caught a bug in someone else's code (below).

## What we built

Nothing here is a new algorithm: the time complexity is Myers'. The gains
are constant factors:

1. **Exact trimming.** d(xPy, xQy) = d(P, Q), so shared prefixes and
   suffixes can be removed first. Near-duplicate pairs shrink to their
   differing core.
2. **Several strings per register.** A 128-bit NEON register holds 8
   patterns of 16 characters (or 4 × 32, 2 × 64), an idea from Hyyrö,
   Fredriksson and Navarro (2005).
3. **Four register groups at once.** Each Myers step depends on the
   previous one, so a single group leaves the CPU waiting. Interleaving four
   independent groups alone gave 1.5-2.5x.
4. **No Python conversion.** A C extension reads Python strings in place;
   in Python, conversion cost more than the computation.
5. **GPU kernels.** For pairs, one thread per pair with the pattern in
   registers. For search, one block per query with its lookup table in
   shared memory.

## Results

**One Apple M4 core, search (one query against many candidates),
millions of pairs per second:**

| | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| Best library | 17.1 | 10.5 | 4.0 | 1.30 | 0.48 |
| Ours | **159.6** | **41.4** | **5.5** | 1.12 | 0.34 |

At length 16, one core beats StringZilla running on all 10 cores (140 M/s).
Above 64 characters we lose: one long pattern fills a whole NEON register.
For independent pairs, timed from Python lists, we are 2.9x the best library
at L16, 2.1x at L32 and 1.9x at L64. One caveat: RapidFuzz has no vector
code for ARM, so part of this lead is a missing port. On an AMD server core,
where RapidFuzz has AVX2, its short-string search is faster than ours.

**NVIDIA H100, our GPU search vs StringZilla's GPU backend** (both returning
results to Python), millions of pairs per second, two runs:

| | L16 | L32 | L64 | L128 | L256 |
|---|---|---|---|---|---|
| StringZilla GPU | 250 / 640 | 750 / 730 | 470 / 500 | 260 / 260 | 120 / 120 |
| Ours | **510 / 1,920** | **1,880 / 2,030** | **1,480 / 1,500** | **840 / 890** | **330 / 340** |

That is 2-3.5x at every length, against a library running at its published
speed on the card its authors benchmark on.

**GPU vs CPU on the same machine.** The GPU computes 100-700x faster than
the best CPU result. Whether that survives depends on data movement. Search
turns 4,000 strings into 4 million results, so it stays 4-119x ahead end to
end. Independent pairs need their own strings moved for every result: at
length 16, just copying the strings into one buffer costs more time than a
CPU core needs to compute the distance. There, the GPU is no faster than
one good CPU core.

## Things we did not expect

- **The VM lied a little.** A NEON kernel that lost in a Linux VM won by
  about 2x natively on the same M4.
- **"All cores" on a rented pod means something else.** The container saw
  128 hardware threads but was allowed 13.6 CPUs.
- **StringZilla's GPU backend has two issues.** On an RTX 4090 every call
  fails with `CUDA_ERROR_INVALID_VALUE`. On an H100, one call in one run
  returned 28 wrong distances out of 300; our correctness gate caught it.
  We have drafted both as issues with a reproduction script.
- **The real-world win was the threshold.** Most users ask "is the distance
  ≤ k?". Stopping early when a pair can no longer get under k is up to 6x
  faster on longer strings, which is more than most of the low-level tuning
  gave.

## Use it

```bash
pip install "bitlev @ git+https://github.com/karims/editdist-bench"
```

```python
import bitlev
bitlev.cdist(queries, choices, max_dist=2, dtype="uint8")
```

`bitlev.gpu` offers the same calls on an NVIDIA GPU. Its limits: score only
(no alignment), unit costs, byte-level strings, and single-threaded on the
CPU.

## What is next

The obvious extensions are a NEON path for 65-128 characters, an AVX2 kernel
for x86 servers, and returning only the matches under a threshold from the
GPU. The protocol, the data generator and the harness are in the repository
for anyone who wants to check our numbers or beat them.
