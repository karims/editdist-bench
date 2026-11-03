"""The C kernels must reproduce ground truth exactly."""

import random

import numpy as np
import pytest

from bench import cmyers
from bench.reference import dp_distance, myers_distance

pytestmark = pytest.mark.skipif(not cmyers.available(), reason="run `make` first")

BOUNDARY = [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 513]


def _rand(rng, n, alpha):
    return "".join(rng.choice(alpha) for _ in range(n))


def _check_pairs(A, B):
    got = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
    exp = np.array([myers_distance(a, b) for a, b in zip(A, B)])
    bad = np.nonzero(got != exp)[0]
    assert bad.size == 0, [(A[i], B[i], int(got[i]), int(exp[i])) for i in bad[:3]]


def test_known_values():
    A = ["", "", "abc", "kitten", "flaw", "intention", "abc"]
    B = ["", "abc", "", "sitting", "lawn", "execution", "cba"]
    got = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
    assert got.tolist() == [0, 3, 3, 3, 2, 5, 2]


@pytest.mark.parametrize("n", BOUNDARY)
def test_word_boundaries(n):
    rng = random.Random(n)
    A, B = [], []
    for _ in range(30):
        A.append(_rand(rng, n, "ACGT"))
        B.append(_rand(rng, rng.randint(0, n + 70), "ACGT"))
    _check_pairs(A, B)
    _check_pairs(B, A)          # pattern/text swap path


def test_random_mixed_lengths_and_alphabets():
    rng = random.Random(99)
    A, B = [], []
    for _ in range(3000):
        alpha = rng.choice(["AB", "ACGT", "abcdefghijklmnopqrstuvwxyz ", "\x00\x01\xff"])
        A.append(_rand(rng, rng.randint(0, 300), alpha))
        B.append(_rand(rng, rng.randint(0, 300), alpha))
    _check_pairs(A, B)


def test_against_dp_small():
    rng = random.Random(3)
    A = [_rand(rng, rng.randint(0, 20), "ab") for _ in range(500)]
    B = [_rand(rng, rng.randint(0, 20), "ab") for _ in range(500)]
    got = cmyers.pairs(cmyers.pack(A), cmyers.pack(B))
    assert got.tolist() == [dp_distance(a, b) for a, b in zip(A, B)]


def test_cross_matches_pairs():
    rng = random.Random(4)
    Q = [_rand(rng, rng.randint(0, 200), "ACGT") for _ in range(25)]
    C = [_rand(rng, rng.randint(0, 200), "ACGT") for _ in range(31)]
    got = cmyers.cross(cmyers.pack(Q), cmyers.pack(C))
    exp = np.array([[myers_distance(q, c) for c in C] for q in Q])
    assert got.shape == (25, 31)
    assert np.array_equal(got, exp)
