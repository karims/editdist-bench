# Pre-registered hypotheses

Written before any benchmark numbers were collected. Each hypothesis states a
falsifiable prediction and the decision it drives. Outcomes are appended below
each one later, in separate commits, without editing the original text.

## Phase 2 — state of the art (prior-art libraries only)

**H2.1 — Bit-parallel libraries dominate.** For L ≤ 64, RapidFuzz and
StringZilla are each ≥ 10× faster per pair than a per-call edlib loop driven
from Python, because per-call overhead dominates at these lengths.
*Decision:* if false, our Python-level comparison is not measuring what we
think, and the harness is fixed before anything else.

**H2.2 — No single library wins every bucket.** At least one bucket is won by a
different library than the overall winner.
*Decision:* if one library wins everything, it is the sole target to beat.

**H2.3 — Short-string throughput is overhead-bound.** At L = 16 the best
single-thread throughput is well below one pair per ~100 CPU cycles (i.e. below
~30 M pairs/s on a ~3 GHz core), leaving room for batching.
*Decision:* if the best library is already within 2× of ~100 cycles/pair at
L ≤ 64, the short-string niche is closed and we move to the GPU phase.

## Phase 3 — plain C Myers (to be measured next)

**H3.1** A plain single-pair C implementation of Myers' algorithm, called once
per batch (not per pair) from Python, is within 2× of the best library at
L ≤ 64. *Decision:* establishes whether the libraries' advantage is algorithmic
or engineering.

## Phase 4 — batched SIMD (NEON)

**H4.1** Packing several independent short pairs into one 128-bit NEON register
(inter-pair parallelism) gives ≥ 1.5× over the best library for L ≤ 64,
single thread. *Decision:* this is the project's main claim; if it fails
after reasonable effort, we report the negative result and move to the GPU.

## Outcomes

### H3.1 — preliminary (Linux VM on Apple M4, 4 vCPUs; quick grid, `bytes` alphabet)

Source: `results/vm-linux/prior_art_20261103-080405.csv`. Not official until
repeated on native macOS.

- **Kernel-only: holds.** Plain C Myers is 0.71-1.41x the best library for
  pairs and 1.04-1.39x for cross, i.e. within 2x everywhere. The libraries'
  speed comes from the algorithm, not from engineering we lack.
- **End-to-end from Python lists: we lose on short pairs** (0.18-0.56x at
  L <= 64) because converting Python strings to a byte buffer costs
  160-400 ns/pair, more than the distance itself. RapidFuzz avoids this by
  reading Python strings in C. Cross is unaffected (conversion is per string,
  not per pair): 1.01-1.37x.
- **Observation:** RapidFuzz is faster at low error rates (L16: 22.6M pairs/s at
  5% vs 14.1M at 20%); our kernel is error-rate independent. Likely cause:
  stripping the common prefix/suffix before the bit-parallel loop. Cheap, exact,
  and worth an ablation.

### Phase 3b — preliminary (Linux VM on Apple M4; `bytes`/`dna`/`text`)

Sources: `results/vm-linux/prior_art_20261103-170443.csv` (L16/64/256 at
0.01/0.2; its L16-e0.01 bucket is degenerate, all pairs identical) and
`results/vm-linux/prior_art_20261103-170548.csv` (L16/32 at 0.01 after the
one-edit fix).

End-to-end from Python lists, single thread, `ours.c-myers-ext` (in-place str
+ prefix/suffix trimming) vs the best library (RapidFuzz `cpdist` everywhere):

| Bucket | Best library | Ours | Ratio |
|---|---|---|---|
| L16, 1 edit (all alphabets) | 22.2-22.8M | 71.8-75.4M | 3.2-3.4x |
| L32, 1 edit (all alphabets) | 19.5-19.7M | 47.2-48.3M | 2.4-2.5x |
| L16, 20% (3 edits) | 13.9M | 32.3M | 2.3x |
| L64, 1% (1 edit) | 15.6M | 28.6M | 1.8x |
| L64, 20% | 4.66M | 4.83M | 1.04x |
| L256, 1% | 1.01M | 1.23M | 1.2x |
| L256, 20% | 0.48M | 0.61M | 1.3x |

Attribution (kernel-only ablation, same runs): trimming alone takes L16/20%
from 19.8M to 33.9M pairs/s and L64/1% from 4.8M to 29.5M; at L64/20% it
changes nothing (edits are spread out, little common affix). Reading str in
place removes the 160-400 ns/pair conversion cost.

Interpretation: these wins are engineering (per-pair overhead, a known exact
shortcut), not a new algorithm. With few edits the trimmed core is tiny and
per-pair overhead dominates; RapidFuzz spends ~40 ns/pair there, we ~14 ns.
Where real DP work remains (L64/20%) we are at parity, which is the regime
phase 4 (inter-pair SIMD) has to win.

### H4.1 — preliminary, cross shape only (Linux VM on Apple M4)

Source: `results/vm-linux/prior_art_20261104-034554.csv` (bytes, 200 x 200,
single thread, kernel-only; env shows git_dirty because the contender
registration was not yet committed when the run started).

| Length | Best library | Scalar ours | NEON ours | NEON / library | NEON / scalar |
|---|---|---|---|---|---|
| 16 | 26.7M | 29.1M | 122M | 4.6x | 4.2x |
| 32 | 12.3M | 12.9M | 23-24M | 1.9-2.0x | 1.8-1.9x |
| 64 | 3.9-4.6M | 5.5-5.8M | 6.0M | 1.3-1.5x | 1.05-1.08x |
| 128 | 1.5M | 1.8M | 1.8M (scalar fallback) | 1.2x | 1.0x |

- **Holds for L <= 32** (>= 1.5x by a wide margin); **borderline at L = 64**
  (1.32x at 5% errors fails, 1.53x at 20% passes); not attempted above 64.
- Caveat on the comparison: RapidFuzz has SIMD for this shape only on x86
  (SSE2/AVX2); on ARM it runs scalar. Part of the margin is that missing port,
  so the claim is "fastest measured on ARM", not "faster than RapidFuzz's
  algorithm".
- NEON gains are far below lane count (4.2x with 8 lanes, 1.05x with 2).
  Likely cause: each Myers step depends on the previous one, so one register
  group is latency-bound. Next: interleave several independent groups.

### Phase 4 step 1 — interleaving independent groups (VM, preliminary)

Source: `results/vm-linux/prior_art_20261104-035621.csv` (bytes, 200 x 200,
single thread, kernel-only, cross shape).

| Length | Best library | NEON K=1 | K=2 | K=4 | K=4 / library |
|---|---|---|---|---|---|
| 16 | 26.9M | 122M | 155-158M | 202-207M | 7.5-7.7x |
| 32 | 12.4-12.5M | 23-24M | 36-38M | 56-58M | 4.5-4.6x |
| 64 | 3.9-4.6M | 6.0-6.1M | 7.8-8.0M | 9.1-9.5M | 2.0-2.3x |

The latency explanation holds: K=4 is 1.5-2.4x over K=1 at every length, so
one register group was waiting on its own dependency chain. With K=4, **H4.1
now holds for the cross shape at every L <= 64** (>= 2x over the best library
on this platform). Same caveat as before: RapidFuzz has no SIMD path on ARM.

### Phase 4 step 2 — NEON kernel for independent pairs (VM, preliminary): negative

Sources: `results/vm-linux/prior_art_20261104-035849.csv` (L16/32) and
`prior_art_20261104-035904.csv` (L64/128); bytes, 10,000 pairs, single
thread, kernel-only, pairs shape. All correctness gates: 0 mismatches.

| Bucket | Best library | Scalar + trim | NEON K=1 | NEON K=4 | K=4 / trim |
|---|---|---|---|---|---|
| L16 e0.05 | 20.0M | 75.8M | 51.7M | 54.3M | 0.72x |
| L16 e0.20 | 13.9M | 29.6M | 9.2M | 9.4M | 0.32x |
| L32 e0.05 | 11.5M | 19.8M | 7.9M | 7.7M | 0.39x |
| L32 e0.20 | 8.6M | 11.4M | 5.2M | 4.9M | 0.43x |
| L64 e0.05 | 6.6M | 7.8M | 3.7M | 3.8M | 0.48x |
| L64 e0.20 | 4.7M | 4.9M | 2.4M | 2.7M | 0.54x |
| L128 e0.05 | 1.6M | 2.3M | 2.5M | 2.5M | 1.08x |
| L128 e0.20 | 1.3M | 1.9M | 1.9M | 1.9M | 1.02x |

- **H4.1 fails for the pairs shape** with this design. The NEON kernel is
  slower than our own scalar kernel with trimming everywhere except L128,
  where it is roughly equal.
- K=4 is about the same as K=1, so this kernel is not latency-bound (unlike
  cross). It is bound by the per-lane gather: in pairs mode every lane has a
  different text, so each step loads one Peq word per lane from memory
  instead of broadcasting one shared character as in cross mode.
- Replacing the store-to-array-then-vector-load gather with direct lane
  inserts (`vsetq_lane`) gave about +5-15% (L16 e0.20: 9.4M -> 10.8M); still
  0.34-0.55x of scalar + trim below L128. Kept, since it is strictly better.
- Trimming also removes most of the work on near-identical pairs, which the
  scalar path exploits pair by pair; lanes in a SIMD group run to the longest
  remaining text, so they waste that saving.
- Decision: the default for pairs stays scalar + trim. Options left open:
  a transposed Peq layout or NEON `tbl` lookups for small alphabets, and a
  hybrid dispatch that uses NEON only for patterns > 64.

### Native macOS (Apple M4) — three runs, supersedes VM numbers where they differ

Sources: `results/macos-m4/prior_art_20261103-232923`, `-233524`,
`-234330` (quick grid, bytes, single thread unless noted). All gates pass in
all runs. Ranges span the runs.

| Result | Ours | Best library (1 thread) | Ratio |
|---|---|---|---|
| Cross L16, NEON K=4 | 128-148M | StringZilla 18.5-18.8M | 7-8x (also >= StringZilla on all 10 cores, 96-138M) |
| Cross L64, NEON K=4 | 5.1-5.3M | StringZilla 3.0-3.8M | 1.4-1.7x |
| Pairs L16 e0.05, scalar + trim (ext) | 49-52M | RapidFuzz 12.4-15.5M | 3.3-4.0x |
| Pairs L64, NEON pairs (ext-neon) | 5.2-8.3M | RapidFuzz 2.7-3.9M | ~2x |
| L256, both shapes | 0.3-0.4M | 0.3-0.5M | ~parity (no NEON path above 64) |

- **H4.1, cross:** holds at L16 (and L32 on the VM); **fails at L64 on native
  hardware** (1.4-1.7x vs the 2x target; the VM showed 2.0-2.3x).
- **Pairs NEON, revised:** the VM negative result does not carry over. On the
  M4 the NEON pairs kernel beats scalar + trim at L64 (e.g. 7.9M vs 5.2M) and
  is ~2x RapidFuzz. At L16 it is roughly level with scalar + trim. The VM's
  slower per-lane gathers made it look worse than it is.
- Caveat unchanged: RapidFuzz has no SIMD path on ARM.

### Native macOS (Apple M4) — full grid, official run 1 of 2

Source: `results/macos-m4/prior_art_20261104-103035` (official script,
20,000 pairs, 200 x 200 cross, 11 interleaved reps). 1,260 rows, all gates
pass, 0 mismatches. Ratios = best of ours / best library, single thread.
Median row IQR/median 9.4%. Needs a second run before claims (PROTOCOL 6.1).

Cross (kernel), same pattern in all three alphabets (bytes / dna / text):

| L | ours / best library |
|---|---|
| 16 | 8.2-9.9x |
| 32 | 4.1-5.0x |
| 64 | 1.5-1.8x |
| 128 | 0.9-1.0x (StringZilla's single thread wins narrowly) |
| 256 | 0.7-0.8x (StringZilla wins) |

Pairs (end-to-end, from Python lists), all three alphabets:

| L | ours / best library |
|---|---|
| 16 | 2.1-3.1x |
| 32 | 2.0-2.6x |
| 64 | 1.8-2.2x |
| 128 | 1.4-2.0x |
| 256 | 1.2-1.4x |

- H4.1 (>= 2x over the best library, L <= 64): **cross holds at L16-32,
  fails at L64 (1.5-1.8x). Pairs holds at L16-32, borderline at L64.**
- Alphabet makes little difference (the Peq table hides it).
- Single-threaded at L16, ours (156M/s) beats StringZilla on all 10 cores
  (114M/s); from L32 on, the multi-threaded library is faster.
- Error rate barely matters for cross; for pairs, low error rates gain most
  (trimming removes more).
