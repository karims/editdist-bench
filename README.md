# editdist-bench

Fast exact edit distance (unit-cost Levenshtein) for **batches of short strings**,
developed with an explicit, reproducible protocol.

The goal is not "a fast edit distance" (many exist) but a measured answer to a
narrow question:

> Can bit-parallel algorithms (Myers / Hyyrö) be made meaningfully faster than
> the current state of the art for **many short pairs (≤ 256 characters)**,
> first on ARM/NEON (Apple Silicon), later on NVIDIA GPUs?

## Status

| Phase | What | State |
|---|---|---|
| 0 | Problem definition and pre-registered hypotheses | done (see `docs/`) |
| 1 | Dataset generator, reference implementations, tests | done |
| 2 | Prior-art benchmark (RapidFuzz, StringZilla, edlib) | runner done, **numbers pending (run on target hardware)** |
| 3 | Plain C Myers baseline | done (VM, preliminary): kernel parity with libraries |
| 3b | Prefix/suffix trimming + in-place `str` extension | done (VM, preliminary): up to 3.4x end-to-end on near-identical short pairs, parity at L64/20% |
| 4 | Batched / SIMD (NEON) variants | next |
| 5 | GPU (CUDA) variant | later |

## Layout

```
docs/PROTOCOL.md     problem definition, buckets, metrics, measurement rules
docs/HYPOTHESES.md   what we expect, written BEFORE measuring (pre-registration)
bench/datagen.py     deterministic pair generator (lengths x error rates x alphabets)
bench/reference.py   ground truth: textbook DP + independent bit-parallel Myers
bench/baselines.py   adapters for prior-art libraries, each behind a correctness gate
bench/envinfo.py     records hardware/OS/compiler/library versions with every result
bench/run_prior_art.py  benchmark runner -> results/*.csv
tests/               property tests and cross-checks
results/             raw benchmark output (committed, never edited by hand)
```

## Quick start

```bash
./setup.sh                                  # venv (Python >= 3.10), deps, C build, tests
source .venv/bin/activate
python -m bench.run_prior_art --quick       # ~ a few minutes
python -m bench.run_prior_art               # full grid
```

Python ≥ 3.10 is required: older Pythons resolve to StringZilla 4.x, whose
`LevenshteinDistances` computes elementwise pairs instead of the 5.x
query × candidate matrix (the adapter detects either, but 5.x is the baseline
of record).

Run benchmarks on mains power, with other apps closed. Results are written to
`results/prior_art_<timestamp>.csv` together with an environment JSON.
Commit result files as they are; never edit them by hand.

Each timed sample repeats the batch until it lasts ≥ 50 ms (the repeat count is
recorded as `inner_repeats`), so sub-millisecond batches are not timer noise.

Only results from the target hardware (Apple Silicon for phases 2–4) count.
A smoke run on a 2-vCPU x86 cloud VM was used to check the harness and is not
recorded as a result.
