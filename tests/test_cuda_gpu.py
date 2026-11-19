"""GPU kernels against the CPU implementation and the reference (needs a GPU)."""
import random

import numpy as np
import pytest

from bench import cmyers, cudamyers
from bench.datagen import Bucket, make_cross, make_pairs
from bench.reference import myers_distance

pytestmark = pytest.mark.skipif(not cudamyers.available(), reason="no NVIDIA GPU / CuPy")


def gpu_pairs(A, B, trim=True):
    st = cudamyers.PairsState(cmyers.pack(A), cmyers.pack(B))
    return cudamyers.pairs(st, trim).get()


def gpu_cross(Q, C):
    st = cudamyers.CrossState(cmyers.pack(Q), cmyers.pack(C))
    return cudamyers.cross(st).get().reshape(len(Q), len(C))


def test_edge_cases():
    A = ["", "", "abc", "a", "kitten", "\x00", "\xff" * 16, "".join(map(chr, range(256))), "x" * 300]
    B = ["", "abc", "", "b", "sitting", "\x00\x00", "\xff" * 15 + "\x00",
         "".join(map(chr, range(255, -1, -1))), "y" * 290]
    exp = [myers_distance(a, b) for a, b in zip(A, B)]
    for trim in (True, False):
        assert list(gpu_pairs(A, B, trim)) == exp
        assert list(gpu_pairs(B, A, trim)) == exp


@pytest.mark.parametrize("m", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_class_boundaries_mixed_batch(m):
    rng = random.Random(m)
    A = ["".join(chr(rng.randrange(256)) for _ in range(rng.randrange(1, m + 1))) for _ in range(300)]
    B = ["".join(chr(rng.randrange(256)) for _ in range(rng.randrange(0, m + 30))) for _ in range(300)]
    exp = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
    np.testing.assert_array_equal(gpu_pairs(A, B), exp)


@pytest.mark.parametrize("L,e,alph", [(16, 0.05, "bytes"), (32, 0.2, "dna"), (64, 0.5, "text"),
                                      (128, 0.2, "bytes"), (256, 0.05, "dna"), (256, 0.5, "bytes")])
def test_benchmark_data(L, e, alph):
    P = make_pairs(Bucket(L, e, alph), 3000, 5)
    A, B = [p[0] for p in P], [p[1] for p in P]
    np.testing.assert_array_equal(gpu_pairs(A, B), cmyers.pairs(cmyers.pack(A), cmyers.pack(B)))
    Q, C = make_cross(Bucket(L, e, alph), 70, 600, 5)       # 600 candidates: >2 thread blocks
    np.testing.assert_array_equal(gpu_cross(Q, C), cmyers.cross(cmyers.pack(Q), cmyers.pack(C)))


def test_cross_long_and_empty_queries():
    Q = ["", "a" * 64, "b" * 65, "c" * 512, "d" * 513, "abc"]
    C = ["", "aaa", "b" * 70, "c" * 500, "xyz"]
    np.testing.assert_array_equal(gpu_cross(Q, C), cmyers.cross(cmyers.pack(Q), cmyers.pack(C)))


@pytest.mark.parametrize("k", [0, 2, 10])
def test_threshold_and_byte_output(k):
    P = make_pairs(Bucket(64, 0.2, "dna"), 2000, 9) + make_pairs(Bucket(256, 0.5, "bytes"), 300, 9)
    A, B = [p[0] for p in P] + ["x" * 300], [p[1] for p in P] + ["y" * 280]   # one CPU-fallback pair
    exact = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
    st = cudamyers.PairsState(cmyers.pack(A), cmyers.pack(B))
    np.testing.assert_array_equal(cudamyers.pairs(st, True, k).get(), np.minimum(exact, k + 1))
    got8 = cudamyers.pairs(st, True, -1, True).get()
    assert got8.dtype == np.uint8
    np.testing.assert_array_equal(got8, np.minimum(exact, 255))
    Q, C = make_cross(Bucket(32, 0.2, "bytes"), 40, 700, 9)
    Q = Q + ["q" * 600]                                                          # CPU-fallback query
    ex = cmyers.cross(cmyers.pack(Q), cmyers.pack(C)).ravel()
    cs = cudamyers.CrossState(cmyers.pack(Q), cmyers.pack(C))
    np.testing.assert_array_equal(cudamyers.cross(cs, k).get(), np.minimum(ex, k + 1))
    g8 = cudamyers.cross(cs, k, True).get()
    assert g8.dtype == np.uint8
    np.testing.assert_array_equal(g8, np.minimum(np.minimum(ex, k + 1), 255))
