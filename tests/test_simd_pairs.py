"""NEON kernel for independent pairs: every lane has its own pattern and text."""

import random

import numpy as np
import pytest

from bench import cmyers
from bench.reference import myers_distance

pytestmark = pytest.mark.skipif(not (cmyers.available() and cmyers.ext_available()),
                                reason="run `make PY=<venv python>` first")


def _rand(rng, n, alpha):
    return "".join(rng.choice(alpha) for _ in range(n))


def _check(A, B):
    exp = np.array([myers_distance(a, b) for a, b in zip(A, B)])
    pa, pb = cmyers.pack(A), cmyers.pack(B)
    for trim in (False, True):
        for k in (1, 2, 4):
            got = cmyers.pairs_simd(pa, pb, trim=trim, k=k)
            bad = np.nonzero(got != exp)[0]
            assert bad.size == 0, (trim, k, [(A[i], B[i], int(got[i]), int(exp[i])) for i in bad[:3]])
            got = cmyers.ext_pairs_simd(A, B, trim=trim, k=k)
            assert np.array_equal(got, exp), ("ext", trim, k)


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 16, 31, 33, 64, 65, 100])
def test_partial_groups(n):
    rng = random.Random(n)
    A = [_rand(rng, rng.randint(1, 70), "ACGT") for _ in range(n)]
    B = [_rand(rng, rng.randint(0, 90), "ACGT") for _ in range(n)]
    _check(A, B)


def test_wildly_different_text_lengths_in_one_class():
    rng = random.Random(1)
    A, B = [], []
    for _ in range(200):
        A.append(_rand(rng, rng.randint(1, 16), "ab"))         # all in the 16-bit class
        B.append(_rand(rng, rng.choice([0, 1, 3, 16, 50, 300]), "ab"))
    _check(A, B)


def test_class_boundaries_both_orders():
    rng = random.Random(2)
    A, B = [], []
    for m in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 120]:
        for n in [0, 1, 15, 16, 17, 32, 33, 64, 65, 130]:
            A.append(_rand(rng, m, "xyz"))
            B.append(_rand(rng, n, "xyz"))
    _check(A, B)
    _check(B, A)


def test_near_duplicates_and_identical():
    rng = random.Random(3)
    A, B = [], []
    for _ in range(300):
        a = _rand(rng, rng.randint(0, 80), "ACGT")
        b = list(a)
        for _ in range(rng.randint(0, 3)):
            if b:
                b[rng.randrange(len(b))] = rng.choice("ACGT")
        A.append(a)
        B.append("".join(b))
    _check(A, B)


def test_all_byte_values():
    rng = random.Random(4)
    alpha = "".join(chr(i) for i in range(256))
    A = [_rand(rng, rng.randint(0, 70), alpha) for _ in range(150)]
    B = [_rand(rng, rng.randint(0, 70), alpha) for _ in range(150)]
    _check(A, B)


def test_long_text_with_short_pattern_skips_16bit_lanes():
    rng = random.Random(5)
    A = [_rand(rng, 10, "ab") for _ in range(9)]
    B = [_rand(rng, 70000, "ab")] + [_rand(rng, 20, "ab") for _ in range(8)]
    exp = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))     # verified scalar kernel
    for k in (1, 2, 4):
        assert np.array_equal(cmyers.pairs_simd(cmyers.pack(A), cmyers.pack(B), k=k), exp)


def test_agrees_with_scalar_on_benchmark_data():
    from bench.datagen import Bucket, make_pairs
    for L in (16, 32, 64, 128):
        for e in (0.01, 0.2, 0.5):
            P = make_pairs(Bucket(L, e, "text"), 400, seed=6)
            A, B = [p[0] for p in P], [p[1] for p in P]
            ref = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
            for k in (1, 2, 4):
                assert np.array_equal(cmyers.ext_pairs_simd(A, B, k=k), ref)


def test_invalid_k():
    with pytest.raises(ValueError):
        cmyers.pairs_simd(cmyers.pack(["a"]), cmyers.pack(["b"]), k=3)
    with pytest.raises(ValueError):
        cmyers.ext_pairs_simd(["a"], ["b"], k=0)
