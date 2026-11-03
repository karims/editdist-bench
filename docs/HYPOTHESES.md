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
