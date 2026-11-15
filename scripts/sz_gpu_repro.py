"""Minimal, self-contained reproductions for two StringZilla CUDA issues.

Needs only: pip install stringzilla stringzillas-cuda numpy  (and an NVIDIA GPU)

  python sz_gpu_repro.py basic     # issue 1: every call fails on some GPUs (seen on RTX 4090)
  python sz_gpu_repro.py stress    # issue 2: compare GPU vs CPU results over many calls
"""
import random
import sys

import numpy as np
import stringzilla as sz
import stringzillas as szs

print("stringzillas capabilities:", szs.__capabilities__)
gpu = szs.DeviceScope(gpu_device=0)
cpu = szs.DeviceScope(cpu_cores=1)
eng_gpu = szs.LevenshteinDistances(capabilities=gpu)
eng_cpu = szs.LevenshteinDistances(capabilities=cpu)


def basic():
    for label, A, B in [("abc vs abd", ["abc"], ["abd"]),
                        ("2 x 2", ["abc", "kitten"], ["abd", "sitting"])]:
        try:
            r = np.asarray(eng_gpu(szs.to_device(sz.Strs(A)), szs.to_device(sz.Strs(B)), device=gpu))
            print(f"{label:12s} OK   {r.ravel().tolist()}")
        except Exception as e:
            print(f"{label:12s} FAIL {e}")


def mutate(s, k, rng):
    s = list(s)
    for _ in range(k):
        op, i = rng.randrange(3), rng.randrange(len(s) + 1)
        if op == 0 and i < len(s): s[i] = rng.choice("ACGT")
        elif op == 1: s.insert(i, rng.choice("ACGT"))
        elif i < len(s): del s[i]
    return "".join(s)


def stress(rounds=2000):
    """Same shape as the call that returned wrong distances for us: 15 queries
    x 20 candidates, DNA, length ~64, ~3 edits, inputs moved with to_device
    inside each call."""
    rng = random.Random(0)
    bases = ["".join(rng.choice("ACGT") for _ in range(64)) for _ in range(4)]
    bad_calls = 0
    for it in range(rounds):
        Q = [mutate(rng.choice(bases), 3, rng) for _ in range(15)]
        C = [mutate(rng.choice(bases), 3, rng) for _ in range(20)]
        want = np.asarray(eng_cpu(sz.Strs(Q), sz.Strs(C), device=cpu)).ravel()
        got = np.asarray(eng_gpu(szs.to_device(sz.Strs(Q)), szs.to_device(sz.Strs(C)), device=gpu)).ravel()
        wrong = int((got != want).sum())
        if wrong:
            bad_calls += 1
            i = int(np.nonzero(got != want)[0][0])
            print(f"call {it}: {wrong}/{want.size} wrong; e.g. Q[{i // 20}] vs C[{i % 20}]: "
                  f"gpu={got[i]} cpu={want[i]}")
    print(f"{bad_calls} of {rounds} calls returned wrong distances")


{"basic": basic, "stress": stress}[sys.argv[1] if len(sys.argv) > 1 else "basic"]()
