"""Ground-truth implementations (PROTOCOL §4).

Two independent algorithms that must always agree:
  dp_distance     textbook O(n*m) Wagner-Fischer dynamic programming
  myers_distance  Myers' bit-parallel algorithm (Hyyro's formulation for edit
                  distance), using Python's arbitrary-precision integers as the
                  bit vectors, so it works for any length with no word limits.

Neither is fast. They exist to be obviously correct and to document the
algorithm the optimized kernels implement.
"""

from __future__ import annotations


def dp_distance(a: str, b: str) -> int:
    """Wagner-Fischer, two rows, O(len(a)*len(b)) time."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                # deletion
                         cur[j - 1] + 1,             # insertion
                         prev[j - 1] + (ca != cb))   # substitution / match
        prev = cur
    return prev[-1]


def myers_distance(a: str, b: str) -> int:
    """Bit-parallel edit distance (Myers 1999; Hyyro 2001 global variant).

    The pattern `a` occupies the bit vectors (bit i <-> a[i]); the text `b` is
    scanned one character at a time. Vertical deltas of the DP column are
    encoded as two bit vectors: VP (+1) and VN (-1). `score` tracks the bottom
    cell D[m][j]. Each text character updates the whole column in O(1) word
    operations (here: big-int operations).
    """
    m = len(a)
    if m == 0:
        return len(b)
    mask = (1 << m) - 1
    high = 1 << (m - 1)

    peq: dict[str, int] = {}
    for i, c in enumerate(a):
        peq[c] = peq.get(c, 0) | (1 << i)

    vp, vn, score = mask, 0, m
    for c in b:
        eq = peq.get(c, 0)
        xv = eq | vn
        xh = ((((eq & vp) + vp) & mask) ^ vp) | eq
        hp = vn | (~(xh | vp) & mask)
        hn = vp & xh
        if hp & high:
            score += 1
        elif hn & high:
            score -= 1
        # Shift horizontal deltas down one row; the top row of a global
        # alignment always increases by 1 per text character (carry-in 1).
        hp = ((hp << 1) | 1) & mask
        hn = (hn << 1) & mask
        vp = hn | (~(xv | hp) & mask)
        vn = hp & xv
    return score
