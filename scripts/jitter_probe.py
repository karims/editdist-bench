"""Where does timing spread come from on this machine? (~30 s)

Times one fixed scalar kernel call 300 times (~20 ms each) and prints the
distribution. A tight single peak means frequency jitter; two separated peaks
(fast ~1x, slow ~1.5-2x) means the thread is being moved between performance
and efficiency cores.

  python scripts/jitter_probe.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import cmyers                                       # noqa: E402
from bench.datagen import Bucket, make_pairs                   # noqa: E402
from bench.run_prior_art import _prefer_performance_cores      # noqa: E402

print("thread QoS:", _prefer_performance_cores())
P = make_pairs(Bucket(64, 0.2, "bytes"), 4000, 0)
pa, pb = cmyers.pack([p[0] for p in P]), cmyers.pack([p[1] for p in P])
f = lambda: cmyers.pairs(pa, pb, trim=False)                   # noqa: E731
t0 = time.perf_counter(); f(); once = time.perf_counter() - t0
inner = max(1, int(0.02 / once))
for _ in range(20):                                            # 0.4 s warm-up
    f()
walls, cpus = [], []
for _ in range(300):
    c0, t0 = time.thread_time(), time.perf_counter()
    for _ in range(inner):
        f()
    walls.append(time.perf_counter() - t0); cpus.append(time.thread_time() - c0)

lo = min(walls)
rel = sorted(w / lo for w in walls)
q = lambda p: rel[int(p * (len(rel) - 1))]                     # noqa: E731
print(f"samples 300 x {inner} calls; relative to fastest sample:")
print(f"  p10 {q(.1):.3f}  p25 {q(.25):.3f}  p50 {q(.5):.3f}  p75 {q(.75):.3f}  "
      f"p90 {q(.9):.3f}  max {rel[-1]:.3f}")
print(f"  IQR/median {(q(.75) - q(.25)) / q(.5):.1%}   wall/CPU median "
      f"{statistics.median(w / c for w, c in zip(walls, cpus)):.3f}")
print("histogram (relative time):")
edges = [1.00, 1.02, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 99]
for a, b in zip(edges, edges[1:]):
    n = sum(a <= x < b for x in rel)
    print(f"  {a:4.2f}-{'' if b == 99 else f'{b:4.2f}':4s} {n:4d} {'#' * (n // 3)}")
print("order (first 60 samples, . = within 5% of fastest, o = 5-35%, X = slower):")
print("  " + "".join("." if w / lo < 1.05 else "o" if w / lo < 1.35 else "X" for w in walls[:60]))
