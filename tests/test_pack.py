import numpy as np
import pytest

from bench import cmyers
from bench.datagen import Bucket, make_pairs

pytestmark = pytest.mark.skipif(not cmyers.ext_available(), reason="extension not built")

CASES = [[], [""], ["", ""], ["ab", "", "xyz"], ["\xff\x00\x80"], ["q" * 1000] * 3,
         [p[0] for p in make_pairs(Bucket(64, 0.2, "bytes"), 500, 1)]]


@pytest.mark.parametrize("S", CASES)
def test_pack_fast_matches_pack(S):
    b1, o1 = cmyers.pack(S)
    b2, o2 = cmyers.pack_fast(S)
    np.testing.assert_array_equal(o1, o2)
    assert bytes(b1[:o1[-1]]) == bytes(b2[:o2[-1]])
    assert len(b2) >= 1                      # never an empty buffer (GPU needs a pointer)


def test_pack_fast_accepts_bytes_and_rejects_wide_str():
    b, o = cmyers.pack_fast([b"by", "tes"])
    assert bytes(b[:o[-1]]) == b"bytes" and list(o) == [0, 2, 5]
    with pytest.raises(ValueError):
        cmyers.pack_fast(["ok", "ā"])
    with pytest.raises(TypeError):
        cmyers.pack_fast(["ok", 3])
