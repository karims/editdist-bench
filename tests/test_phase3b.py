"""Trimming and the in-place str/bytes extension must not change any result."""

import random

import numpy as np
import pytest

from bench import cmyers
from bench.reference import myers_distance

pytestmark = pytest.mark.skipif(not (cmyers.available() and cmyers.ext_available()),
                                reason="run `make PY=<venv python>` first")


def _cases(seed=1):
    rng = random.Random(seed)
    A, B = [], []
    for n in [0, 1, 2, 5, 31, 63, 64, 65, 127, 128, 129, 256, 300]:
        for _ in range(20):
            a = "".join(rng.choice("ACGT") for _ in range(n))
            kind = rng.randrange(4)
            if kind == 0:     # unrelated
                b = "".join(rng.choice("ACGT") for _ in range(rng.randint(0, n + 5)))
            elif kind == 1:   # few edits: long shared prefix/suffix
                b = list(a)
                for _ in range(rng.randint(0, 3)):
                    if b:
                        b[rng.randrange(len(b))] = rng.choice("ACGT")
                b = "".join(b)
            elif kind == 2:   # one string a prefix/suffix of the other
                b = a[: rng.randint(0, n)] if rng.random() < 0.5 else a[rng.randint(0, n):]
            else:             # identical
                b = a
            A.append(a)
            B.append(b)
    A += ["abcab", "aaaa", "abab", "", "x"]
    B += ["abab", "aa", "baba", "", ""]
    return A, B


def test_trim_kernel_matches_reference():
    A, B = _cases()
    exp = [myers_distance(a, b) for a, b in zip(A, B)]
    for trim in (False, True):
        got = cmyers.pairs(cmyers.pack(A), cmyers.pack(B), trim=trim)
        assert got.tolist() == exp, f"trim={trim}"


@pytest.mark.parametrize("trim", [False, True])
def test_ext_pairs_str_and_bytes(trim):
    A, B = _cases(2)
    exp = [myers_distance(a, b) for a, b in zip(A, B)]
    assert cmyers.ext_pairs(A, B, trim).tolist() == exp
    Ab, Bb = [a.encode() for a in A], [b.encode() for b in B]
    assert cmyers.ext_pairs(Ab, Bb, trim).tolist() == exp


def test_ext_latin1_bytes_are_compared_bytewise():
    A, B = ["caf\xe9", "\xff\x00"], ["cafe", "\x00\xff"]
    assert cmyers.ext_pairs(A, B).tolist() == [1, 2]


def test_ext_cross():
    rng = random.Random(3)
    Q = ["".join(rng.choice("ab ") for _ in range(rng.randint(0, 150))) for _ in range(17)]
    C = ["".join(rng.choice("ab ") for _ in range(rng.randint(0, 150))) for _ in range(23)]
    exp = np.array([[myers_distance(q, c) for c in C] for q in Q])
    assert np.array_equal(cmyers.ext_cross(Q, C), exp)


def test_ext_rejects_wide_strings_and_bad_input():
    with pytest.raises(ValueError):
        cmyers.ext_pairs(["日本"], ["日"])
    with pytest.raises(TypeError):
        cmyers.ext_pairs([1], ["a"])
    with pytest.raises(ValueError):
        cmyers._EXT.pairs(["a", "b"], ["a"], np.empty(2, np.int32), True)
    with pytest.raises(ValueError):
        cmyers._EXT.pairs(["a", "b"], ["a", "b"], np.empty(1, np.int32), True)
