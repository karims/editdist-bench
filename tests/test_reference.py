import random

import pytest

from bench.datagen import Bucket, grid, make_pairs
from bench.reference import dp_distance, myers_distance

IMPLS = [dp_distance, myers_distance]

KNOWN = [
    ("", "", 0),
    ("", "abc", 3),
    ("abc", "", 3),
    ("kitten", "sitting", 3),
    ("flaw", "lawn", 2),
    ("intention", "execution", 5),
    ("abc", "abc", 0),
    ("abc", "cba", 2),
    ("a" * 70, "b" * 70, 70),
]


@pytest.mark.parametrize("f", IMPLS)
@pytest.mark.parametrize("a,b,d", KNOWN)
def test_known_values(f, a, b, d):
    assert f(a, b) == d


def _rand(rng, n, alpha="ACGT"):
    return "".join(rng.choice(alpha) for _ in range(n))


@pytest.mark.parametrize("n", [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200])
def test_myers_matches_dp_at_word_boundaries(n):
    rng = random.Random(n)
    for _ in range(40):
        a = _rand(rng, n)
        b = _rand(rng, rng.randint(0, n + 10))
        assert myers_distance(a, b) == dp_distance(a, b)
        assert myers_distance(b, a) == dp_distance(a, b)


def test_myers_matches_dp_random_alphabets():
    rng = random.Random(7)
    for _ in range(1500):
        alpha = rng.choice(["AB", "ACGT", "abcdefghijklmnopqrstuvwxyz "])
        a = _rand(rng, rng.randint(0, 80), alpha)
        b = _rand(rng, rng.randint(0, 80), alpha)
        assert myers_distance(a, b) == dp_distance(a, b)


def test_properties():
    rng = random.Random(11)
    for _ in range(300):
        a, b, c = (_rand(rng, rng.randint(0, 40)) for _ in range(3))
        dab = myers_distance(a, b)
        assert myers_distance(a, a) == 0
        assert dab == myers_distance(b, a)                                  # symmetry
        assert dab <= myers_distance(a, c) + myers_distance(c, b)           # triangle
        assert abs(len(a) - len(b)) <= dab <= max(len(a), len(b))           # bounds
        assert (dab == 0) == (a == b)


def test_generated_pairs_respect_edit_budget():
    for bucket in grid(quick=False):
        for a, b in make_pairs(bucket, 20, seed=3):
            assert len(a) == bucket.length
            d = myers_distance(a, b)
            assert d <= bucket.edits          # edits can cancel, never exceed


def test_generation_is_deterministic():
    b = Bucket(64, 0.2, "text")
    assert make_pairs(b, 50, seed=1) == make_pairs(b, 50, seed=1)
    assert make_pairs(b, 50, seed=1) != make_pairs(b, 50, seed=2)
