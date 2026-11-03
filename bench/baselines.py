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
        out += _stringzilla_contenders()
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

    # ---------------- ours: plain C Myers (phase 3) ----------------
    from bench import cmyers
    if cmyers.available():
        both = lambda A, B: (cmyers.pack(A), cmyers.pack(B))
        out += [
            Contender("ours.c-myers-plain", "pairs", "1", both,
                      lambda st: cmyers.pairs(*st),
                      note="phase 3: one pair at a time, no SIMD, one call per batch"),
            Contender("ours.c-myers-plain", "cross", "1", both,
                      lambda st: cmyers.cross(*st),
                      note="phase 3: pattern loaded once per query"),
        ]
        out.append(Contender("ours.c-myers-trim", "pairs", "1", both,
                             lambda st: cmyers.pairs(*st, trim=True),
                             note="phase 3b ablation: plain + common prefix/suffix trimming"))
    else:
        missing.append("ours.c-myers-plain (run `make`)")
    if cmyers.ext_available():
        ident = lambda A, B: (A, B)
        out += [
            Contender("ours.c-myers-ext", "pairs", "1", ident,
                      lambda st: cmyers.ext_pairs(st[0], st[1], True),
                      note="phase 3b: trim + reads Python str in place (no conversion)"),
            Contender("ours.c-myers-ext", "cross", "1", ident,
                      lambda st: cmyers.ext_cross(st[0], st[1]),
                      note="phase 3b: reads Python str in place"),
        ]
    else:
        missing.append("ours.c-myers-ext (run `make PY=<venv python>`)")

    return out, missing


def stringzilla_semantics() -> str:
    """StringZilla changed the meaning of LevenshteinDistances between major
    versions: 4.x returns elementwise distances for equal-length lists
    ("pairwise"), 5.x returns the full queries x candidates matrix ("cross").
    Detect it from behaviour rather than trusting version numbers."""
    import stringzilla as sz
    import stringzillas as szs
    engine = szs.LevenshteinDistances(capabilities=szs.DeviceScope(cpu_cores=1))
    try:
        r = np.asarray(engine(sz.Strs(["ab", "cd"]), sz.Strs(["ab", "xx", "cd"])))
        if r.shape == (2, 3):
            return "cross"
    except Exception:
        pass
    r = np.asarray(engine(sz.Strs(["ab", "cd"]), sz.Strs(["ab", "xx"])))
    if r.shape == (2,):
        return "pairwise"
    raise RuntimeError(f"unrecognised StringZilla LevenshteinDistances output shape {r.shape}")


def _stringzilla_contenders() -> list[Contender]:
    import stringzilla as sz
    import stringzillas as szs

    mode = stringzilla_semantics()
    out = []
    for cores, threads in ((1, "1"), (0, "all")):
        scope = szs.DeviceScope(cpu_cores=cores)
        engine = szs.LevenshteinDistances(capabilities=scope)
        call = (lambda e, s: (lambda st: e(st[0], st[1], device=s)))(engine, scope)

        if mode == "cross":
            out.append(Contender("stringzilla.LevenshteinDistances", "cross", threads,
                                 lambda A, B: (sz.Strs(A), sz.Strs(B)), call))
        else:
            out.append(Contender("stringzilla.LevenshteinDistances", "pairs", threads,
                                 lambda A, B: (sz.Strs(A), sz.Strs(B)), call))
            # Cross shape through the pairwise API: expand Q x C into flat lists
            # (expansion happens in prepare, i.e. untimed, but costs memory).
            out.append(Contender(
                "stringzilla.LevenshteinDistances(expanded)", "cross", threads,
                lambda A, B: (sz.Strs([a for a in A for _ in B]), sz.Strs(list(B) * len(A))),
                call, note="4.x pairwise API fed an expanded Q*C list"))

    if mode == "cross":
        # 5.x has no pairwise API: a 1x1 matrix per pair is the only exact option.
        scope1 = szs.DeviceScope(cpu_cores=1)
        engine1 = szs.LevenshteinDistances(capabilities=scope1)
        out.append(Contender(
            "stringzilla.1x1-loop", "pairs", "1",
            lambda A, B: [(sz.Strs([a]), sz.Strs([b])) for a, b in zip(A, B)],
            lambda st: [int(np.asarray(engine1(x, y, device=scope1)).ravel()[0]) for x, y in st],
            note="5.x has no pairwise API; one 1x1 matrix call per pair"))
    return out


def as_vector(result) -> np.ndarray:
    return _flat(result)
