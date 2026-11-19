"""The bitlev library API against the reference implementation (CPU).
Runs against an installed bitlev, or the in-place build (`pip install -e .`)."""
import random

import numpy as np
import pytest

import bitlev                                              # noqa: E402
from bench.datagen import Bucket, make_cross, make_pairs   # noqa: E402
from bench.reference import myers_distance as ref           # noqa: E402


def _data(seed=0, n=400):
    rng = random.Random(seed)
    A, B = [], []
    for _ in range(n):
        L = rng.choice([0, 1, 7, 16, 31, 40, 64, 65, 100, 200, 300])
        a = "".join(rng.choice("ACGTx\x00\xff") for _ in range(L))
        b = list(a)
        for _ in range(rng.randrange(0, 10)):
            i, op = rng.randrange(len(b) + 1), rng.randrange(3)
            if op == 0 and i < len(b): b[i] = rng.choice("ACGT")
            elif op == 1: b.insert(i, rng.choice("ACGT"))
            elif i < len(b): del b[i]
        A.append(a)
        B.append("".join(b) if rng.random() < 0.8 else "".join(rng.choice("ACGT") for _ in range(rng.randrange(70))))
    return A, B


def test_distance_examples():
    assert bitlev.distance("kitten", "sitting") == 3
    assert bitlev.distance("", "abc") == 3
    assert bitlev.distance(b"abc", b"abd") == 1
    assert bitlev.distance("kitten", "sitting", max_dist=1) == 2


@pytest.mark.parametrize("method", ["auto", "scalar"] + (["simd"] if bitlev.simd_available() else []))
@pytest.mark.parametrize("k", [None, 0, 1, 3, 20])
def test_pairwise(method, k):
    A, B = _data(1)
    exp = np.array([ref(a, b) for a, b in zip(A, B)])
    if k is not None:
        exp = np.minimum(exp, k + 1)
    got = bitlev.pairwise(A, B, max_dist=k, method=method)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize("method", ["auto", "scalar", "simd"])
@pytest.mark.parametrize("k", [None, 0, 2, 15])
def test_cdist(method, k):
    A, B = _data(2, 120)
    Q, C = A[:40], B[:70]
    exp = np.array([[ref(x, y) for y in C] for x in Q])
    if k is not None:
        exp = np.minimum(exp, k + 1)
    got = bitlev.cdist(Q, C, max_dist=k, method=method)
    assert got.shape == (40, 70)
    np.testing.assert_array_equal(got, exp)


def test_dtypes_saturate():
    Q, C = ["a" * 300, "b"], ["c" * 10, "b" * 400]
    exact = bitlev.cdist(Q, C)
    for dt, cap in (("uint8", 255), (np.uint16, 65535)):
        got = bitlev.cdist(Q, C, dtype=dt)
        assert got.dtype == np.dtype(dt)
        np.testing.assert_array_equal(got, np.minimum(exact, cap))
    assert bitlev.pairwise(["x" * 300], [""], dtype=np.uint8)[0] == 255


def test_benchmark_data_and_iterables():
    P = make_pairs(Bucket(32, 0.2, "text"), 500, 4)
    A, B = (p[0] for p in P), (p[1] for p in P)          # generators are accepted
    got = bitlev.pairwise(A, B)
    np.testing.assert_array_equal(got, [ref(a, b) for a, b in P])
    Q, C = make_cross(Bucket(16, 0.05, "dna"), 20, 30, 4)
    np.testing.assert_array_equal(bitlev.cdist(tuple(Q), C), [[ref(x, y) for y in C] for x in Q])


def test_errors_and_empty():
    with pytest.raises(ValueError):
        bitlev.pairwise(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        bitlev.distance("ok", "ā")                   # above U+00FF: encode first
    with pytest.raises(TypeError):
        bitlev.distance("ok", 3)
    with pytest.raises(ValueError):
        bitlev.cdist(["a"], ["b"], max_dist=-1)
    with pytest.raises(ValueError):
        bitlev.cdist(["a"], ["b"], dtype=np.float32)
    assert bitlev.pairwise([], []).shape == (0,)
    assert bitlev.cdist([], ["a"]).shape == (0, 1)
    assert bitlev.cdist(["a"], []).shape == (1, 0)
