# Draft note for RapidFuzz (GitHub Discussions or an issue on rapidfuzz/RapidFuzz)

**Title:** Interest in an ARM NEON path for batch Levenshtein (`cdist` / `cpdist`)?

Hi Max, and thanks for RapidFuzz: it was the main baseline in a small
benchmarking project of mine,
https://github.com/karims/editdist-bench.

On Apple Silicon (M4) we found that `process.cdist` / `cpdist` with
`Levenshtein` run without vector instructions; as far as we could tell the
batch SIMD paths cover SSE2/AVX2 only. A NEON version of the same idea
(several short patterns per 128-bit register, 8 × 16-bit, 4 × 32-bit or
2 × 64-bit lanes, plus interleaving a few independent register groups to
hide latency) measured on one M4 core, 200 × 200 cdist, random bytes, 5%
edits:

| Length | RapidFuzz `cdist` | NEON kernel |
|---|---|---|
| 16 | 17.1 M pairs/s | 159.6 M pairs/s |
| 32 | 8.2 M pairs/s | 41.4 M pairs/s |
| 64 | 2.7 M pairs/s | 5.5 M pairs/s |

Raw data, environment and protocol are in the repo (results/macos-m4/).
The kernel is plain C with NEON intrinsics (csrc/myers.c,
`DEFINE_KGROUP_KERNEL`).

Would a NEON path in rapidfuzz-cpp be welcome? If so, I'm happy to help
port it in whatever shape fits your SIMD abstractions, or to share anything
from the benchmark that's useful.
