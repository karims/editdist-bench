"""The NEON cross kernel must match ground truth on every class and edge."""

import random

import numpy as np
import pytest

from bench import cmyers
from bench.reference import myers_distance

pytestmark = pytest.mark.skipif(not cmyers.available(), reason="run `make` first")

EDGE = [0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 130]


def _rand(rng, n, alpha):
    return "".join(rng.choice(alpha) for _ in range(n))


def _check(Q, C):
    exp = np.array([[myers_distance(q, c) for c in C] for q in Q]).reshape(len(Q), len(C))
    for k in (1, 2, 4):
        got = cmyers.cross_simd(cmyers.pack(Q), cmyers.pack(C), k=k)
        bad = np.argwhere(got != exp)
        assert bad.size == 0, (k, [(Q[i], C[j], int(got[i, j]), int(exp[i, j])) for i, j in bad[:3]])


def test_reports_neon_status():
    # informative: on ARM builds this must be True
    import platform
    if platform.machine() in ("aarch64", "arm64"):
        assert cmyers.simd_available()


@pytest.mark.parametrize("n_cands", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40])
def test_partial_groups(n_cands):
    rng = random.Random(n_cands)
    for m in (5, 16, 30, 64):
        C = [_rand(rng, m, "ACGT") for _ in range(n_cands)]
        Q = [_rand(rng, rng.randint(0, 80), "ACGT") for _ in range(6)]
        _check(Q, C)


def test_class_boundaries_and_empties():
    rng = random.Random(1)
    C = [_rand(rng, m, "ab") for m in EDGE for _ in range(3)]
    Q = [_rand(rng, n, "ab") for n in EDGE]
    _check(Q, C)


def test_mixed_classes_shuffled_and_all_bytes():
    rng = random.Random(2)
    alpha = "".join(chr(i) for i in range(256))
    C = [_rand(rng, rng.randint(0, 140), alpha) for _ in range(60)]
    Q = [_rand(rng, rng.randint(0, 140), alpha) for _ in range(25)]
    _check(Q, C)


def test_near_duplicates():
    rng = random.Random(3)
    base = [_rand(rng, rng.randint(1, 64), "ACGT") for _ in range(10)]
    C = []
    for b in base:
        for _ in range(3):
            s = list(b)
            s[rng.randrange(len(s))] = rng.choice("ACGT")
            C.append("".join(s))
    _check(base, C)


def test_long_query_disables_16bit_lanes():
    rng = random.Random(4)
    C = [_rand(rng, rng.randint(1, 16), "ab") for _ in range(9)]
    Q = [_rand(rng, 70000, "ab"), "ab", ""]
    ref = cmyers.cross(cmyers.pack(Q), cmyers.pack(C))   # scalar kernel, already verified
    for k in (1, 2, 4):
        assert np.array_equal(cmyers.cross_simd(cmyers.pack(Q), cmyers.pack(C), k=k), ref)


def test_agrees_with_scalar_on_benchmark_like_data():
    from bench.datagen import Bucket, make_cross
    for L in (16, 32, 64, 128):
        Q, C = make_cross(Bucket(L, 0.2, "text"), 40, 50, seed=5)
        ref = cmyers.cross(cmyers.pack(Q), cmyers.pack(C))
        for k in (1, 2, 4):
            assert np.array_equal(cmyers.cross_simd(cmyers.pack(Q), cmyers.pack(C), k=k), ref)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        cmyers.cross_simd(cmyers.pack(["a"]), cmyers.pack(["b"]), k=3)
