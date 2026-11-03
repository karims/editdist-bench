"""Deterministic generation of string pairs for the workload grid (PROTOCOL §2)."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

# English letter frequencies (percent) plus space; used for the `text` alphabet.
_EN_FREQ = {
    "e": 12.7, "t": 9.1, "a": 8.2, "o": 7.5, "i": 7.0, "n": 6.7, "s": 6.3,
    "h": 6.1, "r": 6.0, "d": 4.3, "l": 4.0, "c": 2.8, "u": 2.8, "m": 2.4,
    "w": 2.4, "f": 2.2, "g": 2.0, "y": 2.0, "p": 1.9, "b": 1.5, "v": 1.0,
    "k": 0.8, "j": 0.15, "x": 0.15, "q": 0.1, "z": 0.07,
}
_SPACE_SHARE = 0.18  # roughly one space per ~5.5 characters

ALPHABETS = ("dna", "bytes", "text")
LENGTHS = (16, 32, 64, 128, 256)
ERROR_RATES = (0.01, 0.05, 0.20, 0.50)

QUICK_LENGTHS = (16, 64, 256)
QUICK_ERROR_RATES = (0.05, 0.20)
QUICK_ALPHABETS = ("bytes",)


@dataclass(frozen=True)
class Bucket:
    length: int
    error_rate: float
    alphabet: str

    @property
    def name(self) -> str:
        return f"{self.alphabet}-L{self.length}-e{self.error_rate:g}"

    @property
    def edits(self) -> int:
        # At least one edit: otherwise short low-error buckets (e.g. L16 at 1%)
        # contain only identical pairs and measure duplicate detection instead.
        return max(1, round(self.error_rate * self.length))


def grid(quick: bool = False) -> list[Bucket]:
    lengths = QUICK_LENGTHS if quick else LENGTHS
    errors = QUICK_ERROR_RATES if quick else ERROR_RATES
    alphas = QUICK_ALPHABETS if quick else ALPHABETS
    return [Bucket(L, e, a) for a in alphas for L in lengths for e in errors]


class _Sampler:
    def __init__(self, alphabet: str, rng: random.Random):
        self.rng = rng
        if alphabet == "dna":
            self.chars, self.weights = list("ACGT"), None
        elif alphabet == "bytes":
            self.chars, self.weights = list(string.ascii_letters + string.digits), None
        elif alphabet == "text":
            letters = list(_EN_FREQ)
            total = sum(_EN_FREQ.values())
            w = [f / total * (1 - _SPACE_SHARE) for f in _EN_FREQ.values()]
            self.chars, self.weights = letters + [" "], w + [_SPACE_SHARE]
        else:
            raise ValueError(f"unknown alphabet {alphabet!r}")

    def string(self, n: int) -> str:
        return "".join(self.rng.choices(self.chars, weights=self.weights, k=n))

    def char(self) -> str:
        return self.rng.choices(self.chars, weights=self.weights, k=1)[0]


def mutate(a: str, edits: int, sampler: _Sampler) -> str:
    """Apply `edits` random substitutions/insertions/deletions to `a`."""
    rng = sampler.rng
    s = list(a)
    for _ in range(edits):
        op = rng.randrange(3) if s else 1  # only insertion possible on empty
        if op == 0:  # substitution (always changes the character if possible)
            i = rng.randrange(len(s))
            c = sampler.char()
            for _ in range(8):
                if c != s[i]:
                    break
                c = sampler.char()
            s[i] = c
        elif op == 1:  # insertion
            s.insert(rng.randrange(len(s) + 1), sampler.char())
        else:  # deletion
            del s[rng.randrange(len(s))]
    return "".join(s)


def make_pairs(bucket: Bucket, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """n pairs for a bucket; deterministic in (bucket, n, seed)."""
    rng = random.Random(f"{seed}|{bucket.name}")
    sampler = _Sampler(bucket.alphabet, rng)
    out = []
    for _ in range(n):
        a = sampler.string(bucket.length)
        out.append((a, mutate(a, bucket.edits, sampler)))
    return out


def make_cross(bucket: Bucket, n_queries: int, n_candidates: int,
               seed: int = 0) -> tuple[list[str], list[str]]:
    """Queries and candidates for the cross shape.

    Candidates are mutated copies of a few base strings, and queries are mutated
    copies of the same bases, so the matrix mixes near and far pairs (like a
    real fuzzy-search dictionary) instead of being uniformly random.
    """
    rng = random.Random(f"{seed}|cross|{bucket.name}")
    sampler = _Sampler(bucket.alphabet, rng)
    n_bases = max(1, n_candidates // 20)
    bases = [sampler.string(bucket.length) for _ in range(n_bases)]
    cands = [mutate(rng.choice(bases), bucket.edits, sampler) for _ in range(n_candidates)]
    queries = [mutate(rng.choice(bases), bucket.edits, sampler) for _ in range(n_queries)]
    return queries, cands
