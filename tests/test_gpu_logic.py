"""The GPU kernels' device functions, compiled for the CPU and checked against
the reference implementations. Runs everywhere; no GPU needed."""
import ctypes
import os
import random

import pytest

from bench import cmyers
from bench.datagen import Bucket, make_cross, make_pairs
from bench.reference import dp_distance, myers_distance

PATH = os.path.join(os.path.dirname(cmyers.LIB_PATH), "libgpucheck.so")
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="run `make` first")


@pytest.fixture(scope="module")
def lib():
    L = ctypes.CDLL(PATH)
    for f in (L.gpu_pair_distance_host, L.gpu_peq_distance_host):
        f.restype = ctypes.c_int
    L.gpu_pair_distance_host.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                         ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    L.gpu_peq_distance_host.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int]
    return L


def swar(L, a: bytes, b: bytes, trim=1):
    return L.gpu_pair_distance_host(a, len(a), b, len(b), trim)


def peq(L, a: bytes, b: bytes):
    return L.gpu_peq_distance_host(a, len(a), b, len(b))


def ref(a: bytes, b: bytes):
    return myers_distance(a.decode("latin-1"), b.decode("latin-1"))


EDGE = [(b"", b""), (b"", b"abc"), (b"abc", b""), (b"a", b"a"), (b"a", b"b"),
        (b"kitten", b"sitting"), (b"\x00", b"\x00\x00"), (b"\x00" * 5, b"\x01" * 3),
        (b"\xff" * 16, b"\xff" * 15 + b"\x00"), (bytes(range(256)), bytes(range(255, -1, -1)))]


@pytest.mark.parametrize("a,b", EDGE)
def test_edge_cases(lib, a, b):
    exp = dp_distance(a.decode("latin-1"), b.decode("latin-1"))
    for t in (0, 1):
        assert swar(lib, a, b, t) == exp
        assert swar(lib, b, a, t) == exp
    if len(a) <= 512:
        assert peq(lib, a, b) == exp


@pytest.mark.parametrize("m", [1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                               127, 128, 129, 255, 256])
def test_class_boundaries(lib, m):
    rng = random.Random(m)
    for _ in range(20):
        a = bytes(rng.randrange(256) for _ in range(m))
        b = bytes(rng.randrange(256) for _ in range(rng.randrange(m, m + 40)))
        e = ref(a, b)
        assert swar(lib, a, b, 0) == e and swar(lib, a, b, 1) == e and swar(lib, b, a, 1) == e
        assert peq(lib, a, b) == e


def test_small_alphabet_and_zero_bytes(lib):
    """Zero bytes matter: the packed pattern is zero-padded."""
    rng = random.Random(7)
    for _ in range(500):
        m = rng.randrange(1, 70)
        a = bytes(rng.choice(b"\x00\x01") for _ in range(m))
        b = bytes(rng.choice(b"\x00\x01") for _ in range(rng.randrange(0, 90)))
        e = ref(a, b)
        assert swar(lib, a, b, 1) == e and swar(lib, a, b, 0) == e
        assert peq(lib, a, b) == e


def test_longer_than_256_is_rejected(lib):
    assert swar(lib, b"a" * 257, b"b" * 300) == -1


@pytest.mark.parametrize("L,e,alph", [(16, 0.05, "bytes"), (32, 0.2, "dna"),
                                      (64, 0.5, "text"), (128, 0.2, "bytes"),
                                      (256, 0.05, "dna")])
def test_benchmark_data(lib, L, e, alph):
    for a, b in make_pairs(Bucket(L, e, alph), 200, 3):
        a, b = a.encode("latin-1"), b.encode("latin-1")
        assert swar(lib, a, b, 1) == ref(a, b)
    Q, C = make_cross(Bucket(L, e, alph), 10, 10, 3)
    for q in Q:
        for c in C:
            q_, c_ = q.encode("latin-1"), c.encode("latin-1")
            assert peq(lib, q_, c_) == ref(q_, c_)
