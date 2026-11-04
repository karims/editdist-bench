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
import random
import statistics
import sys
import time
from datetime import datetime

import numpy as np

from bench import envinfo, sysload
from bench.baselines import as_vector, contenders
from bench.datagen import Bucket, grid, make_cross, make_pairs
from bench.reference import myers_distance

GATE_SAMPLE = 300
FIELDS = ["bucket", "alphabet", "length", "error_rate", "shape", "contender", "threads",
          "n_pairs", "cells", "gate", "mismatches_vs_consensus", "prep_s",
          "median_s", "iqr_s", "inner_repeats", "pairs_per_s", "gcups", "ns_per_pair",
          "e2e_pairs_per_s", "cpu_median_s", "cpu_pairs_per_s", "wall_over_cpu",
          "checksum", "note"]


MIN_SAMPLE_S = 0.05   # each timed sample repeats the batch until it lasts this long
SETTLE_S = 1.0        # idle pause before each thread class, so pools/clocks settle


def _calibrate(fn, state):
    """Warm-up call (its output is the one we check) and the number of inner
    repeats needed for one timed sample to last >= MIN_SAMPLE_S."""
    t0 = time.perf_counter()
    out = fn(state)
    once = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(MIN_SAMPLE_S / once) + 1) if once < MIN_SAMPLE_S else 1
    return out, inner


def _sample(fn, state, inner):
    """One timed sample -> (wall s per batch, thread-CPU s per batch, checksum).

    Thread CPU time excludes intervals when the OS ran something else on this
    core, so wall/CPU > 1 means the sample was disturbed. For multi-threaded
    contenders it only covers the calling thread and is not reported."""
    fn(state)                                  # untimed: re-warm caches after other contenders
    c0, t0 = time.thread_time(), time.perf_counter()
    for _ in range(inner):
        r = fn(state)
    wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
    return wall / inner, cpu / inner, int(as_vector(r).sum())


def _iqr(xs):
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return q[2] - q[0]


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


def _prefer_performance_cores() -> str:
    """macOS: ask for the highest thread QoS so the timing thread is scheduled
    on performance cores rather than drifting onto efficiency cores (a major
    source of run-to-run spread on Apple Silicon). No-op elsewhere."""
    if sys.platform != "darwin":
        return "n/a"
    try:
        import ctypes
        qos_user_interactive = 0x21
        rc = ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(
            qos_user_interactive, 0)
        return "user-interactive" if rc == 0 else f"failed rc={rc}"
    except Exception as e:  # never block a run on this
        return f"failed: {e}"


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
    ap.add_argument("--threads", choices=["1", "all", "both"], default="both",
                    help="only single-threaded, only multi-threaded, or both contender kinds")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to start if the machine is busy, on battery or throttled")
    args = ap.parse_args()

    n_pairs = args.pairs or (5000 if args.quick else 20000)
    n_cross = args.cross or (100 if args.quick else 200)

    all_c, missing = contenders()
    cs = [c for c in all_c if args.only in c.name
          and (args.shape == "both" or c.shape == args.shape)
          and (args.threads == "both" or c.threads == args.threads)]
    if missing:
        print(f"NOTE: not installed, skipped: {', '.join(missing)}", file=sys.stderr)

    before = sysload.snapshot()
    if args.strict:
        # Starting this process (loading Python, numpy and the libraries) itself
        # wakes macOS's code-signing check (syspolicyd), so wait for it to pass.
        for attempt in range(25):
            problems = sysload.strict_problems(before)
            if not problems:
                break
            if attempt == 24:
                print("--strict: not starting:\n  " + "\n  ".join(problems), file=sys.stderr)
                sys.exit(2)
            print("  waiting: " + "; ".join(problems), file=sys.stderr)
            time.sleep(5)
            before = sysload.snapshot()

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env = envinfo.collect()
    env["thread_qos"] = _prefer_performance_cores()
    env.update({"args": vars(args), "n_pairs": n_pairs, "n_cross": n_cross,
                "skipped_missing": missing, "timing": "interleaved-shuffled",
                "sysload_before": before})
    csv_path = os.path.join(args.out, f"prior_art_{stamp}.csv")
    env_path = os.path.join(args.out, f"prior_art_{stamp}.env.json")
    with open(env_path, "w") as f:
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

            live = []                                  # (contender, row, state, prep_s, out, inner)
            for c in group:
                gate = _gate(c, A, B)
                row = {"bucket": bucket.name, "alphabet": bucket.alphabet,
                       "length": bucket.length, "error_rate": bucket.error_rate,
                       "shape": shape, "contender": c.name, "threads": c.threads,
                       "n_pairs": n, "cells": cells, "gate": gate, "note": c.note}
                rows.append(row)
                if gate != "pass":
                    print(f"{bucket.name:22s} {shape:5s} {c.name:34s} {c.threads:>3s}  GATE {gate}",
                          file=sys.stderr)
                    continue
                state = c.prepare(A, B)                       # warm (imports, caches)
                preps = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    state = c.prepare(A, B)
                    preps.append(time.perf_counter() - t0)
                out, inner = _calibrate(c.run, state)
                live.append((c, row, state, statistics.median(preps), out, inner))

            # Interleave: every round times each contender once, in a fresh random
            # order, so a burst of background load spreads across contenders
            # instead of landing on whichever one happened to be running.
            # Single- and multi-threaded contenders are interleaved only within
            # their own class: multi-threaded runs load every core (lowering the
            # clock) and their worker pools may keep spinning after returning,
            # which would disturb single-threaded samples. Settle in between.
            walls = [[] for _ in live]
            cpus = [[] for _ in live]
            sinks = [0] * len(live)
            rng = random.Random(f"{args.seed}:{bucket.name}:{shape}")
            for cls in ("1", "all"):
                idx = [i for i, x in enumerate(live) if x[0].threads == cls]
                if not idx:
                    continue
                time.sleep(SETTLE_S)
                for _ in range(args.reps):
                    rng.shuffle(idx)
                    for i in idx:
                        c, _, state, _, _, inner = live[i]
                        w_, c_, k_ = _sample(c.run, state, inner)
                        walls[i].append(w_); cpus[i].append(c_); sinks[i] += k_

            consensus = None
            for i, (c, row, state, prep_s, out, inner) in enumerate(live):
                med, iqr = statistics.median(walls[i]), _iqr(walls[i])
                vec = as_vector(out)
                if consensus is None:
                    consensus = vec
                mism = int((vec != consensus).sum())
                row.update({
                    "mismatches_vs_consensus": mism,
                    "prep_s": f"{prep_s:.4e}", "median_s": f"{med:.4e}", "iqr_s": f"{iqr:.4e}",
                    "pairs_per_s": f"{n / med:.1f}", "gcups": f"{cells / med / 1e9:.4f}",
                    "ns_per_pair": f"{med / n * 1e9:.1f}",
                    "e2e_pairs_per_s": f"{n / (med + prep_s):.1f}",   # incl. conversion from Python lists
                    "checksum": int(vec.sum()),
                    "inner_repeats": inner,
                })
                if c.threads == "1":
                    cmed = statistics.median(cpus[i])
                    row.update({"cpu_median_s": f"{cmed:.4e}",
                                "cpu_pairs_per_s": f"{n / cmed:.1f}",
                                "wall_over_cpu": f"{med / cmed:.3f}"})
                flag = f"  wall/cpu {med / statistics.median(cpus[i]):.2f}" \
                    if c.threads == "1" and med > 1.05 * statistics.median(cpus[i]) else ""
                print(f"{bucket.name:22s} {shape:5s} {c.name:34s} {c.threads:>3s}  "
                      f"{n / med:>14,.0f} pairs/s  {cells / med / 1e9:7.3f} GCUPS"
                      f"  iqr {iqr / med:4.0%}{flag}"
                      f"{'  MISMATCH ' + str(mism) if mism else ''}", file=sys.stderr)

            with open(csv_path, "w", newline="") as f:      # rewrite after each bucket
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    env["sysload_after"] = after = sysload.snapshot()
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    _summary(rows)
    _noise_report(rows, before, after)
    print(f"\nwrote {csv_path}", file=sys.stderr)


def _noise_report(rows, before, after):
    ok = [r for r in rows if r["gate"] == "pass"]
    if not ok:
        return
    spread = sorted(float(r["iqr_s"]) / float(r["median_s"]) for r in ok)
    woc = sorted(float(r["wall_over_cpu"]) for r in ok if r.get("wall_over_cpu"))
    print(f"\nNoise: IQR/median median {statistics.median(spread):.1%}, "
          f"max {spread[-1]:.1%}, rows > 5%: {sum(x > 0.05 for x in spread)}/{len(spread)}",
          file=sys.stderr)
    if woc:
        print(f"       wall/CPU (1-thread) median {statistics.median(woc):.3f}, "
              f"max {woc[-1]:.3f}  (1.000 = never descheduled)", file=sys.stderr)
    print(f"       other processes' CPU: {before['others_cpu_pct']:.0f}% before, "
          f"{after['others_cpu_pct']:.0f}% after", file=sys.stderr)


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
