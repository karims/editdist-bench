"""Host-side bookkeeping of bench/cudamyers.py (no GPU needed)."""
import numpy as np

from bench import cmyers, cudamyers


def _off(lengths):
    o = np.zeros(len(lengths) + 1, dtype=np.int64)
    o[1:] = np.cumsum(lengths)
    return o


def test_pair_classes_use_shorter_length_and_boundaries():
    la = [0, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 300, 5]
    lb = [9, 99, 17, 32, 400, 64, 65, 999, 129, 256, 257, 10, 300]
    groups, rest = cudamyers.pair_classes(_off(la), _off(lb))
    got = {i: k for k, g in enumerate(groups) for i in g}
    m = [min(x, y) for x, y in zip(la, lb)]
    for i, mm in enumerate(m):
        if mm > 256:
            assert i in rest and i not in got
        else:
            cap = cudamyers.PAIR_CAPS[got[i]]
            assert mm <= cap and (got[i] == 0 or mm > cudamyers.PAIR_CAPS[got[i] - 1])
    assert sorted(list(rest) + list(got)) == list(range(len(la)))


def test_query_classes():
    lq = [0, 1, 64, 65, 128, 129, 256, 257, 512, 513]
    groups, rest = cudamyers.query_classes(_off(lq))
    got = {i: cudamyers.CROSS_NWS[k] for k, g in enumerate(groups) for i in g}
    assert got == {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 8, 8: 8}
    assert list(rest) == [9]


def test_subset_repacks_exactly():
    S = ["ab", "", "xyz", "q" * 300]
    packed = cmyers.pack(S)
    buf, off = cudamyers._subset(packed, [3, 0, 1])
    got = [bytes(buf[off[i]:off[i + 1]]).decode() for i in range(3)]
    assert got == [S[3], S[0], S[1]]
