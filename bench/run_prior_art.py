"""Benchmark prior-art libraries over the workload grid (PROTOCOL §2-§7).

Usage:
  python -m bench.run_prior_art --quick
  python -m bench.run_prior_art --pairs 20000 --cross 200 --reps 5
  python -m bench.run_prior_art --only rapidfuzz --shape pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np

from bench import envinfo
from bench.baselines import as_vector, contenders
from bench.datagen import Bucket, grid, make_cross, make_pairs
from bench.reference import myers_distance

GATE_SAMPLE = 300
FIELDS = ["bucket", "alphabet", "length", "error_rate", "shape", "contender", "threads",
          "n_pairs", "cells", "gate", "mismatches_vs_consensus", "prep_s",
          "median_s", "iqr_s", "inner_repeats", "pairs_per_s", "gcups", "ns_per_pair",
          "e2e_pairs_per_s", "checksum", "note"]


MIN_SAMPLE_S = 0.05   # each timed sample repeats the batch until it lasts this long


def _timed(fn, state, reps):
    """Returns (output, median seconds per batch, IQR seconds per batch, inner)."""
    t0 = time.perf_counter()
    out = fn(state)                       # warm-up (also the output we check)
    once = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(MIN_SAMPLE_S / once) + 1) if once < MIN_SAMPLE_S else 1
    times = []
    sink = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            r = fn(state)
        times.append((time.perf_counter() - t0) / inner)
        sink += int(as_vector(r).sum())   # consume
    q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
    return out, statistics.median(times), q[2] - q[0], inner


def _gate(c, A, B):
    """Compare a sample of the contender's output with independent ground truth."""
    if c.shape == "pairs":
        k = min(GATE_SAMPLE, len(A))
        got = as_vector(c.run(c.prepare(A[:k], B[:k])))
        exp = np.array([myers_distance(a, b) for a, b in zip(A[:k], B[:k])])
    else:
        qa, cb = A[:15], B[:20]
        got = as_vector(c.run(c.prepare(qa, cb)))
        exp = np.array([myers_distance(a, b) for a in qa for b in cb])
    return "pass" if np.array_equal(got, exp) else f"FAIL({int((got != exp).sum())})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pairs", type=int, default=None, help="pairs per bucket")
    ap.add_argument("--cross", type=int, default=None, help="queries = candidates per bucket")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", choices=["pairs", "cross", "both"], default="both")
    ap.add_argument("--only", default="", help="substring filter on contender names")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lengths", default="", help="override grid, e.g. 16,64")
    ap.add_argument("--errors", default="", help="override grid, e.g. 0.01,0.2")
    ap.add_argument("--alphabets", default="", help="override grid, e.g. bytes,dna")
    args = ap.parse_args()

    n_pairs = args.pairs or (5000 if args.quick else 20000)
    n_cross = args.cross or (100 if args.quick else 200)

    all_c, missing = contenders()
    cs = [c for c in all_c if args.only in c.name
          and (args.shape == "both" or c.shape == args.shape)]
    if missing:
        print(f"NOTE: not installed, skipped: {', '.join(missing)}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env = envinfo.collect()
    env.update({"args": vars(args), "n_pairs": n_pairs, "n_cross": n_cross,
                "skipped_missing": missing})
    csv_path = os.path.join(args.out, f"prior_art_{stamp}.csv")
    with open(os.path.join(args.out, f"prior_art_{stamp}.env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(f"cpu: {env.get('cpu')}   git: {env['git_commit'][:8]}"
          f"{' (dirty)' if env['git_dirty'] else ''}", file=sys.stderr)

    rows = []
    buckets = grid(args.quick)
    sel_L = {int(x) for x in args.lengths.split(",")} if args.lengths else None
    sel_E = {float(x) for x in args.errors.split(",")} if args.errors else None
    sel_A = set(args.alphabets.split(",")) if args.alphabets else None
    if sel_L or sel_E or sel_A:
        from bench.datagen import ALPHABETS, ERROR_RATES, LENGTHS
        buckets = [Bucket(L, e, a)
                   for a in (sorted(sel_A) if sel_A else ALPHABETS)
                   for L in (sorted(sel_L) if sel_L else LENGTHS)
                   for e in (sorted(sel_E) if sel_E else ERROR_RATES)]
    for bucket in buckets:
        for shape in ("pairs", "cross"):
            group = [c for c in cs if c.shape == shape]
            if not group:
                continue
            if shape == "pairs":
                P = make_pairs(bucket, n_pairs, args.seed)
                A, B = [p[0] for p in P], [p[1] for p in P]
                n = len(P)
                cells = sum(len(a) * len(b) for a, b in P)
            else:
                A, B = make_cross(bucket, n_cross, n_cross, args.seed)
                n = len(A) * len(B)
                cells = sum(len(a) for a in A) * sum(len(b) for b in B)

            consensus = None
            for c in group:
                gate = _gate(c, A, B)
                row = {"bucket": bucket.name, "alphabet": bucket.alphabet,
                       "length": bucket.length, "error_rate": bucket.error_rate,
                       "shape": shape, "contender": c.name, "threads": c.threads,
                       "n_pairs": n, "cells": cells, "gate": gate, "note": c.note}
                if gate != "pass":
                    rows.append(row)
                    print(f"{bucket.name:22s} {shape:5s} {c.name:34s} {c.threads:>3s}  GATE {gate}",
                          file=sys.stderr)
                    continue
                state = c.prepare(A, B)                       # warm (imports, caches)
                preps = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    state = c.prepare(A, B)
                    preps.append(time.perf_counter() - t0)
                prep_s = statistics.median(preps)
                out, med, iqr, inner = _timed(c.run, state, args.reps)
                vec = as_vector(out)
                if consensus is None:
                    consensus = vec
                mism = int((vec != consensus).sum())
                row.update({
                    "mismatches_vs_consensus": mism,
                    "prep_s": f"{prep_s:.6f}", "median_s": f"{med:.6f}", "iqr_s": f"{iqr:.6f}",
                    "pairs_per_s": f"{n / med:.1f}", "gcups": f"{cells / med / 1e9:.4f}",
                    "ns_per_pair": f"{med / n * 1e9:.1f}",
                    "e2e_pairs_per_s": f"{n / (med + prep_s):.1f}",   # incl. conversion from Python lists
                    "checksum": int(vec.sum()),
                    "inner_repeats": inner,
                })
                rows.append(row)
                print(f"{bucket.name:22s} {shape:5s} {c.name:34s} {c.threads:>3s}  "
                      f"{n / med:>14,.0f} pairs/s  {cells / med / 1e9:7.3f} GCUPS"
                      f"{'  MISMATCH ' + str(mism) if mism else ''}", file=sys.stderr)

            with open(csv_path, "w", newline="") as f:      # rewrite after each bucket
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    _summary(rows)
    print(f"\nwrote {csv_path}", file=sys.stderr)


def _summary(rows):
    ok = [r for r in rows if r["gate"] == "pass"]
    if not ok:
        return
    print("\nBest contender per bucket (pairs/s):")
    keys = sorted({(r["shape"], r["threads"], r["bucket"]) for r in ok})
    for shape, threads, bucket in keys:
        grp = [r for r in ok if (r["shape"], r["threads"], r["bucket"]) == (shape, threads, bucket)]
        best = max(grp, key=lambda r: float(r["pairs_per_s"]))
        print(f"  {shape:5s} t={threads:3s} {bucket:22s} {best['contender']:34s} "
              f"{float(best['pairs_per_s']):>14,.0f}")


if __name__ == "__main__":
    main()
