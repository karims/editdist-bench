"""Adapters for prior-art libraries (PROTOCOL §5).

Each contender exposes:
  shape       "pairs" or "cross"
  threads     "1" or "all"
  prepare(A, B) -> state        untimed conversion into the library's containers
  run(state) -> sequence/array  the timed call; pairs -> N ints, cross -> Q*C ints

Contenders whose library is not installed are skipped (reported, not hidden).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Contender:
    name: str
    shape: str            # "pairs" | "cross"
    threads: str          # "1" | "all"
    prepare: Callable[[list[str], list[str]], Any]
    run: Callable[[Any], Any]
    note: str = ""


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _flat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).ravel()


def contenders() -> tuple[list[Contender], list[str]]:
    out: list[Contender] = []
    missing: list[str] = []

    # ---------------- RapidFuzz ----------------
    if _have("rapidfuzz"):
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein

        def rf_batch(fn, workers):
            return lambda st: fn(st[0], st[1], scorer=Levenshtein.distance,
                                 dtype=np.int32, workers=workers)

        ident = lambda A, B: (A, B)
        out += [
            Contender("rapidfuzz.cpdist", "pairs", "1", ident, rf_batch(process.cpdist, 1)),
            Contender("rapidfuzz.cpdist", "pairs", "all", ident, rf_batch(process.cpdist, -1)),
            Contender("rapidfuzz.distance-loop", "pairs", "1", ident,
                      lambda st: [Levenshtein.distance(a, b) for a, b in zip(*st)],
                      note="one Python call per pair; measures call overhead"),
            Contender("rapidfuzz.cdist", "cross", "1", ident, rf_batch(process.cdist, 1)),
            Contender("rapidfuzz.cdist", "cross", "all", ident, rf_batch(process.cdist, -1)),
        ]
    else:
        missing.append("rapidfuzz")

    # ---------------- StringZilla ----------------
    if _have("stringzilla") and _have("stringzillas"):
        import stringzilla as sz
        import stringzillas as szs

        def sz_cross(cores):
            scope = szs.DeviceScope(cpu_cores=cores)
            engine = szs.LevenshteinDistances(capabilities=scope)

            def prep(A, B):
                return (sz.Strs(A), sz.Strs(B))
            return prep, (lambda st: engine(st[0], st[1], device=scope))

        p1, r1 = sz_cross(1)
        pa, ra = sz_cross(0)
        out += [
            Contender("stringzilla.LevenshteinDistances", "cross", "1", p1, r1),
            Contender("stringzilla.LevenshteinDistances", "cross", "all", pa, ra),
        ]
        # No pairwise API: a 1x1 matrix per pair is the only exact option.
        engine1 = szs.LevenshteinDistances(capabilities=szs.DeviceScope(cpu_cores=1))
        scope1 = szs.DeviceScope(cpu_cores=1)
        out.append(Contender(
            "stringzilla.1x1-loop", "pairs", "1",
            lambda A, B: [(sz.Strs([a]), sz.Strs([b])) for a, b in zip(A, B)],
            lambda st: [int(engine1(x, y, device=scope1)[0, 0]) for x, y in st],
            note="no pairwise API; one 1x1 matrix call per pair"))
    else:
        missing.append("stringzilla/stringzillas")

    # ---------------- edlib ----------------
    if _have("edlib"):
        import edlib

        def ed(a, b):
            return edlib.align(a, b, mode="NW", task="distance")["editDistance"]

        out += [
            Contender("edlib.align-loop", "pairs", "1", lambda A, B: (A, B),
                      lambda st: [ed(a, b) for a, b in zip(*st)],
                      note="one Python call per pair; edlib has no batch API"),
            Contender("edlib.align-loop", "cross", "1", lambda A, B: (A, B),
                      lambda st: [ed(a, b) for a in st[0] for b in st[1]],
                      note="nested Python loop"),
        ]
    else:
        missing.append("edlib")

    return out, missing


def as_vector(result) -> np.ndarray:
    return _flat(result)
