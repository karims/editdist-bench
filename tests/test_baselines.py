"""Every installed contender must reproduce ground truth on tricky inputs."""

import random

import numpy as np
import pytest

from bench.baselines import as_vector, contenders
from bench.reference import myers_distance

CONTENDERS, _ = contenders()


def _cases():
    rng = random.Random(5)
    A, B = [], []
    for n in [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]:
        for _ in range(3):
            a = "".join(rng.choice("ACGTx") for _ in range(n))
            b = "".join(rng.choice("ACGTx") for _ in range(rng.randint(0, n + 5)))
            A.append(a)
            B.append(b)
    A += ["kitten", "", "same"]
    B += ["sitting", "", "same"]
    return A, B


@pytest.mark.parametrize("c", CONTENDERS, ids=lambda c: f"{c.name}[{c.shape},{c.threads}]")
def test_contender_matches_ground_truth(c):
    A, B = _cases()
    if c.shape == "pairs":
        expect = [myers_distance(a, b) for a, b in zip(A, B)]
        got = as_vector(c.run(c.prepare(A, B)))
    else:
        A, B = A[::3], B[::3]
        expect = [myers_distance(a, b) for a in A for b in B]
        got = as_vector(c.run(c.prepare(A, B)))
    assert np.array_equal(got, np.asarray(expect)), c.name
