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
    from bench.sysload import effective_cpus
    ncpu = effective_cpus()        # "all" = the CPUs we are allowed, not os.cpu_count()
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
            Contender("rapidfuzz.cpdist", "pairs", "all", ident, rf_batch(process.cpdist, ncpu)),
            Contender("rapidfuzz.distance-loop", "pairs", "1", ident,
                      lambda st: [Levenshtein.distance(a, b) for a, b in zip(*st)],
                      note="one Python call per pair; measures call overhead"),
            Contender("rapidfuzz.cdist", "cross", "1", ident, rf_batch(process.cdist, 1)),
            Contender("rapidfuzz.cdist", "cross", "all", ident, rf_batch(process.cdist, ncpu)),
        ]
    else:
        missing.append("rapidfuzz")

    # ---------------- StringZilla ----------------
    if _have("stringzilla") and _have("stringzillas"):
        sz_out, sz_missing = _stringzilla_contenders(ncpu)
        out += sz_out
        missing += sz_missing
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
        if cmyers.simd_available():
            out.append(Contender("ours.c-myers-neon", "cross", "1", both,
                                 lambda st: cmyers.cross_simd(*st),
                                 note="phase 4: candidates packed 8/4/2 per NEON register"))
            for kk in (2, 4):
                out.append(Contender(f"ours.c-myers-neon-k{kk}", "cross", "1", both,
                                     (lambda k: lambda st: cmyers.cross_simd(*st, k=k))(kk),
                                     note=f"phase 4: {kk} independent register groups interleaved"))
        if cmyers.simd_available():
            for kk in (1, 4):
                out.append(Contender(f"ours.c-myers-neon-pairs-k{kk}", "pairs", "1", both,
                                     (lambda k: lambda st: cmyers.pairs_simd(*st, trim=True, k=k))(kk),
                                     note=f"phase 4.2: trim + NEON lanes per pair, K={kk}"))
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
            Contender("ours.c-myers-ext-neon" if cmyers.simd_available() else "ours.c-myers-ext-k4-scalar",
                      "pairs", "1", ident,
                      lambda st: cmyers.ext_pairs_simd(st[0], st[1], True, 4),
                      note="phase 4.2: in-place str + trim + NEON pairs, K=4"),
            Contender("ours.c-myers-ext", "cross", "1", ident,
                      lambda st: cmyers.ext_cross(st[0], st[1]),
                      note="phase 3b: reads Python str in place"),
        ]
    else:
        missing.append("ours.c-myers-ext (run `make PY=<venv python>`)")

    # ---------------- GPU (phase 5) ----------------
    from bench import cudamyers
    if cudamyers.available():
        from bench import cmyers as _cm
        packed = lambda A, B: (_cm.pack(A), _cm.pack(B))
        out += [
            Contender("ours.cuda-pairs", "pairs", "gpu",
                      lambda A, B: cudamyers.PairsState(*packed(A, B)),
                      lambda st: cudamyers.pairs(st, True),
                      note="phase 5: kernel-only; inputs on the GPU, result left there"),
            Contender("ours.cuda-pairs+copy", "pairs", "gpu", packed,
                      lambda st: cudamyers.pairs(cudamyers.PairsState(*st), True).get(),
                      note="phase 5: includes copy to GPU, class split and copy back"),
            Contender("ours.cuda-cross", "cross", "gpu",
                      lambda A, B: cudamyers.CrossState(*packed(A, B)),
                      lambda st: cudamyers.cross(st),
                      note="phase 5: kernel-only; inputs on the GPU, result left there"),
            Contender("ours.cuda-cross+copy", "cross", "gpu", packed,
                      lambda st: cudamyers.cross(cudamyers.CrossState(*st)).get(),
                      note="phase 5: includes copy to GPU and copy back"),
        ]
    else:
        missing.append("ours.cuda-* (no NVIDIA GPU or CuPy)")

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


def _stringzilla_contenders(ncpu: int) -> tuple[list[Contender], list[str]]:
    import stringzilla as sz
    import stringzillas as szs

    mode = stringzilla_semantics()
    out = []
    for cores, threads in ((1, "1"), (ncpu, "all")):
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

    missing: list[str] = []
    gpu_ok = False
    if "cuda" in tuple(getattr(szs, "__capabilities__", ())):
        # Probe once: on some GPUs the CUDA backend fails on every input
        # (seen on an RTX 4090 with 5.3.2: CUDA_ERROR_INVALID_VALUE). Record
        # that as unavailable instead of failing the whole run.
        try:
            gscope = szs.DeviceScope(gpu_device=0)
            gengine = szs.LevenshteinDistances(capabilities=gscope)
            r = np.asarray(gengine(szs.to_device(sz.Strs(["abc", "kitten"])),
                                   szs.to_device(sz.Strs(["abd", "sitting"])), device=gscope))
            from bench.reference import myers_distance
            exp = [myers_distance(a, b) for a in ("abc", "kitten") for b in ("abd", "sitting")]
            gpu_ok = r.ravel().tolist() == exp
            if not gpu_ok:
                missing.append(f"stringzilla.LevenshteinDistances[gpu] (wrong probe result {r.ravel().tolist()})")
        except Exception as e:
            missing.append(f"stringzilla.LevenshteinDistances[gpu] (fails on this GPU: {e})")
    if gpu_ok:
        gcall = lambda st: gengine(st[0], st[1], device=gscope)
        shape = "cross" if mode == "cross" else "pairs"
        out.append(Contender(
            "stringzilla.LevenshteinDistances", shape, "gpu",
            lambda A, B: (szs.to_device(sz.Strs(A)), szs.to_device(sz.Strs(B))), gcall,
            note="CUDA backend; inputs already in GPU-accessible (unified) memory"))
        out.append(Contender(
            "stringzilla.LevenshteinDistances+copy", shape, "gpu",
            lambda A, B: (sz.Strs(A), sz.Strs(B)),
            lambda st: gcall((szs.to_device(st[0]), szs.to_device(st[1]))),
            note="CUDA backend; includes moving inputs to GPU-accessible memory"))

    if mode == "cross":
        # 5.x has no pairwise API: a 1x1 matrix per pair is the only exact option.
        scope1 = szs.DeviceScope(cpu_cores=1)
        engine1 = szs.LevenshteinDistances(capabilities=scope1)
        out.append(Contender(
            "stringzilla.1x1-loop", "pairs", "1",
            lambda A, B: [(sz.Strs([a]), sz.Strs([b])) for a, b in zip(A, B)],
            lambda st: [int(np.asarray(engine1(x, y, device=scope1)).ravel()[0]) for x, y in st],
            note="5.x has no pairwise API; one 1x1 matrix call per pair"))
    return out, missing


def as_vector(result) -> np.ndarray:
    if type(result).__module__.startswith("cupy"):
        result = result.get()                     # GPU array -> host
    return _flat(result)
