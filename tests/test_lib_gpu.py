"""bitlev.gpu against the CPU library (needs an NVIDIA GPU and CuPy)."""
import numpy as np
import pytest

import bitlev                                              # noqa: E402
from bitlev import gpu                                      # noqa: E402
from bench.datagen import Bucket, make_cross, make_pairs   # noqa: E402

pytestmark = pytest.mark.skipif(not gpu.available(), reason="no NVIDIA GPU / CuPy")


@pytest.mark.parametrize("k", [None, 0, 3])
@pytest.mark.parametrize("dt", [np.int32, np.uint8, np.uint16])
def test_pairwise_matches_cpu(k, dt):
    P = make_pairs(Bucket(64, 0.2, "bytes"), 3000, 2) + make_pairs(Bucket(256, 0.5, "dna"), 300, 2)
    A, B = [p[0] for p in P] + ["x" * 300, ""], [p[1] for p in P] + ["y" * 290, "abc"]
    np.testing.assert_array_equal(gpu.pairwise(A, B, max_dist=k, dtype=dt),
                                  bitlev.pairwise(A, B, max_dist=k, dtype=dt))


@pytest.mark.parametrize("k", [None, 0, 3])
@pytest.mark.parametrize("dt", [np.int32, np.uint8])
def test_cdist_matches_cpu(k, dt):
    Q, C = make_cross(Bucket(32, 0.2, "text"), 50, 700, 2)
    Q = Q + ["", "q" * 600]
    np.testing.assert_array_equal(gpu.cdist(Q, C, max_dist=k, dtype=dt),
                                  bitlev.cdist(Q, C, max_dist=k, dtype=dt))


def test_return_device_and_empty():
    import cupy as cp
    r = gpu.cdist(["abc"], ["abd", "x"], return_device=True)
    assert isinstance(r, cp.ndarray) and r.get().tolist() == [[1, 3]]
    assert gpu.pairwise([], []).shape == (0,)
    assert gpu.cdist([], ["a"]).shape == (0, 1)
