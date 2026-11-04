/*
 * Plain bit-parallel edit distance (Myers 1999, block formulation as in
 * Hyyro 2001 / edlib). Unit-cost Levenshtein, score only, exact.
 *
 * Phase 3 baseline: deliberately straightforward. One pair at a time, no
 * SIMD, no threading. The point is to learn how much of the libraries' lead
 * is algorithm vs engineering.
 *
 * Pattern = the shorter string, packed into W = ceil(m/64) 64-bit words.
 * For each text character, every block updates its vertical deltas
 * (Pv = +1 bits, Mv = -1 bits) and passes a horizontal delta (-1/0/+1) to
 * the block below. The score at the last pattern row changes by the last
 * block's horizontal delta, read at bit (m-1) % 64.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t word;

/* One block step. hin in {-1,0,+1}; returns hout read at bit `outbit`. */
static inline int block_step(word *Pv_, word *Mv_, word Eq, int hin, int outbit) {
    word Pv = *Pv_, Mv = *Mv_;
    word hin_neg = (word)(hin < 0);
    word Xv = Eq | Mv;
    Eq |= hin_neg;
    word Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;
    word Ph = Mv | ~(Xh | Pv);
    word Mh = Pv & Xh;
    int hout = (int)((Ph >> outbit) & 1) - (int)((Mh >> outbit) & 1);
    Ph = (Ph << 1) | (word)(hin > 0);
    Mh = (Mh << 1) | hin_neg;
    *Pv_ = Mh | ~(Xv | Ph);
    *Mv_ = Ph & Xv;
    return hout;
}

/* Peq workspace: 256 rows of W words, kept all-zero between pattern loads. */
typedef struct { word *peq; int W; } ws_t;

static int ws_init(ws_t *ws, int W) {
    ws->W = W < 1 ? 1 : W;
    ws->peq = (word *)calloc((size_t)256 * ws->W, sizeof(word));
    return ws->peq ? 0 : -1;
}
static void ws_free(ws_t *ws) { free(ws->peq); ws->peq = NULL; }

static void load_pattern(ws_t *ws, const uint8_t *p, int m) {
    for (int i = 0; i < m; i++)
        ws->peq[(size_t)p[i] * ws->W + (i >> 6)] |= (word)1 << (i & 63);
}
static void clear_pattern(ws_t *ws, const uint8_t *p, int m) {
    for (int i = 0; i < m; i++)
        ws->peq[(size_t)p[i] * ws->W + (i >> 6)] = 0;
}

/* Distance between the loaded pattern (length m >= 1) and text t. */
static int run_text(const ws_t *ws, int m, const uint8_t *t, int n, word *Pv, word *Mv) {
    const int W = (m + 63) >> 6;
    const int lastbit = (m - 1) & 63;
    for (int w = 0; w < W; w++) { Pv[w] = ~(word)0; Mv[w] = 0; }
    int score = m;
    if (W == 1) {                                   /* single-word fast path */
        word pv = ~(word)0, mv = 0;
        for (int j = 0; j < n; j++) {
            score += block_step(&pv, &mv, ws->peq[(size_t)t[j] * ws->W], 1, lastbit);
        }
        return score;
    }
    for (int j = 0; j < n; j++) {
        const word *eq = ws->peq + (size_t)t[j] * ws->W;
        int h = 1;                                  /* top row: +1 per column */
        for (int w = 0; w < W - 1; w++) h = block_step(&Pv[w], &Mv[w], eq[w], h, 63);
        score += block_step(&Pv[W - 1], &Mv[W - 1], eq[W - 1], h, lastbit);
    }
    return score;
}

/* Exact: d(xPy, xQy) == d(P, Q) for any common prefix x and suffix y. */
static inline void trim_affix(const uint8_t **a, int *la, const uint8_t **b, int *lb) {
    int n = *la < *lb ? *la : *lb, p = 0;
    while (p < n && (*a)[p] == (*b)[p]) p++;
    int s = 0;
    while (s < n - p && (*a)[*la - 1 - s] == (*b)[*lb - 1 - s]) s++;
    *a += p; *b += p;
    *la -= p + s; *lb -= p + s;
}

/* Distance of one pair; pattern = shorter string. ws must hold >= ceil(min/64) words. */
static int pair_distance(ws_t *ws, word *Pv, word *Mv,
                         const uint8_t *a, int la, const uint8_t *b, int lb, int trim) {
    if (trim) trim_affix(&a, &la, &b, &lb);
    if (la > lb) { const uint8_t *t = a; a = b; b = t; int x = la; la = lb; lb = x; }
    if (la == 0) return lb;
    load_pattern(ws, a, la);
    int d = run_text(ws, la, b, lb, Pv, Mv);
    clear_pattern(ws, a, la);
    return d;
}

static int max_words(const int64_t *off_a, const int64_t *off_b, int64_t n) {
    int64_t mx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t la = off_a[i + 1] - off_a[i], lb = off_b[i + 1] - off_b[i];
        int64_t m = la < lb ? la : lb;
        if (m > mx) mx = m;
    }
    return (int)((mx + 63) >> 6);
}

/* N independent pairs. Strings are concatenated in buf_a/buf_b; pair i is
 * buf[off[i] : off[i+1]]. Returns 0 on success, -1 on allocation failure. */
int ed_pairs_opt(const uint8_t *buf_a, const int64_t *off_a,
                 const uint8_t *buf_b, const int64_t *off_b,
                 int64_t n, int32_t *out, int trim) {
    int W = max_words(off_a, off_b, n);
    ws_t ws;
    if (ws_init(&ws, W)) return -1;
    word *Pv = malloc(sizeof(word) * ws.W), *Mv = malloc(sizeof(word) * ws.W);
    if (!Pv || !Mv) { free(Pv); free(Mv); ws_free(&ws); return -1; }
    for (int64_t i = 0; i < n; i++)
        out[i] = pair_distance(&ws, Pv, Mv,
                               buf_a + off_a[i], (int)(off_a[i + 1] - off_a[i]),
                               buf_b + off_b[i], (int)(off_b[i + 1] - off_b[i]), trim);
    free(Pv); free(Mv); ws_free(&ws);
    return 0;
}

int ed_pairs(const uint8_t *buf_a, const int64_t *off_a,
             const uint8_t *buf_b, const int64_t *off_b,
             int64_t n, int32_t *out) {
    return ed_pairs_opt(buf_a, off_a, buf_b, off_b, n, out, 0);
}

/* Q x C matrix (row-major). Each query is loaded as the pattern once and
 * reused for every candidate - the one-vs-many advantage. */
int ed_cross(const uint8_t *buf_q, const int64_t *off_q, int64_t nq,
             const uint8_t *buf_c, const int64_t *off_c, int64_t nc,
             int32_t *out) {
    int64_t mx = 0;
    for (int64_t i = 0; i < nq; i++) {
        int64_t l = off_q[i + 1] - off_q[i];
        if (l > mx) mx = l;
    }
    ws_t ws;
    if (ws_init(&ws, (int)((mx + 63) >> 6))) return -1;
    word *Pv = malloc(sizeof(word) * ws.W), *Mv = malloc(sizeof(word) * ws.W);
    if (!Pv || !Mv) { free(Pv); free(Mv); ws_free(&ws); return -1; }
    for (int64_t i = 0; i < nq; i++) {
        const uint8_t *q = buf_q + off_q[i];
        int lq = (int)(off_q[i + 1] - off_q[i]);
        int32_t *row = out + i * nc;
        if (lq == 0) {
            for (int64_t j = 0; j < nc; j++) row[j] = (int32_t)(off_c[j + 1] - off_c[j]);
            continue;
        }
        load_pattern(&ws, q, lq);
        for (int64_t j = 0; j < nc; j++)
            row[j] = run_text(&ws, lq, buf_c + off_c[j], (int)(off_c[j + 1] - off_c[j]), Pv, Mv);
        clear_pattern(&ws, q, lq);
    }
    free(Pv); free(Mv); ws_free(&ws);
    return 0;
}

/* ------------------------------------------------------------------------
 * Phase 4: inter-pair SIMD (NEON) for the cross shape.
 *
 * Candidates are packed as patterns, LANES per 128-bit register (16-bit lanes
 * for length <= 16, 32-bit for <= 32, 64-bit for <= 64). The query is the
 * text, shared by all lanes, so every lane advances in lockstep and none
 * finishes early. Each lane's carries stay inside the lane (NEON adds and
 * shifts are per-lane), and each lane reads its score at its own last bit via
 * a per-lane mask. Packing short patterns into one word is due to Hyyro,
 * Fredriksson & Navarro (2005); this is that idea on NEON lanes.
 *
 * Longer candidates (> 64) and builds without NEON use the scalar kernel.
 * ---------------------------------------------------------------------- */

int ed_simd_available(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

#if defined(__ARM_NEON)
#include <arm_neon.h>

/* One group: up to LANES candidate patterns (idx[]) vs every query. */
#define DEFINE_GROUP_KERNEL(NAME, VT, ET, LANES, DUP, LD, ST, AND, ORR, EOR, ADD, SUB, SHL1, TST) \
static void NAME(const uint8_t *buf_q, const int64_t *off_q, int64_t nq,                         \
                 const uint8_t *buf_c, const int64_t *off_c, int64_t nc,                         \
                 const int64_t *idx, int cnt, int32_t *out) {                                     \
    ET table[256][LANES];                                                                         \
    ET last[LANES], m0[LANES];                                                                    \
    memset(table, 0, sizeof(table));                                                              \
    for (int l = 0; l < LANES; l++) {                                                             \
        last[l] = 0; m0[l] = 0;                                                                   \
        if (l >= cnt) continue;                                                                   \
        const uint8_t *c = buf_c + off_c[idx[l]];                                                 \
        int m = (int)(off_c[idx[l] + 1] - off_c[idx[l]]);                                         \
        for (int i = 0; i < m; i++) table[c[i]][l] |= (ET)((ET)1 << i);                           \
        last[l] = (ET)((ET)1 << (m - 1));                                                         \
        m0[l] = (ET)m;                                                                            \
    }                                                                                             \
    const VT ones = DUP((ET)~(ET)0), one = DUP((ET)1);                                            \
    const VT L = LD(last), M0 = LD(m0);                                                           \
    ET res[LANES];                                                                                \
    for (int64_t q = 0; q < nq; q++) {                                                            \
        const uint8_t *t = buf_q + off_q[q];                                                      \
        int64_t n = off_q[q + 1] - off_q[q];                                                      \
        VT Pv = ones, Mv = DUP((ET)0), score = M0;                                                \
        for (int64_t j = 0; j < n; j++) {                                                         \
            VT Eq = LD(table[t[j]]);                                                              \
            VT Xv = ORR(Eq, Mv);                                                                  \
            VT Xh = ORR(EOR(ADD(AND(Eq, Pv), Pv), Pv), Eq);                                       \
            VT Ph = ORR(Mv, EOR(ORR(Xh, Pv), ones));                                              \
            VT Mh = AND(Pv, Xh);                                                                  \
            score = ADD(SUB(score, TST(Ph, L)), TST(Mh, L));   /* tst -> all ones = -1 */          \
            Ph = ORR(SHL1(Ph), one);                           /* top row: +1 per column */       \
            Mh = SHL1(Mh);                                                                        \
            Pv = ORR(Mh, EOR(ORR(Xv, Ph), ones));                                                 \
            Mv = AND(Ph, Xv);                                                                     \
        }                                                                                         \
        ST(res, score);                                                                           \
        for (int l = 0; l < cnt; l++)                                                             \
            out[q * nc + idx[l]] = (m0[l] == 0) ? (int32_t)n : (int32_t)res[l];                   \
    }                                                                                             \
}

#define SHL1_16(x) vshlq_n_u16((x), 1)
#define SHL1_32(x) vshlq_n_u32((x), 1)
#define SHL1_64(x) vshlq_n_u64((x), 1)
DEFINE_GROUP_KERNEL(group16, uint16x8_t, uint16_t, 8, vdupq_n_u16, vld1q_u16, vst1q_u16,
                    vandq_u16, vorrq_u16, veorq_u16, vaddq_u16, vsubq_u16, SHL1_16, vtstq_u16)
DEFINE_GROUP_KERNEL(group32, uint32x4_t, uint32_t, 4, vdupq_n_u32, vld1q_u32, vst1q_u32,
                    vandq_u32, vorrq_u32, veorq_u32, vaddq_u32, vsubq_u32, SHL1_32, vtstq_u32)
DEFINE_GROUP_KERNEL(group64, uint64x2_t, uint64_t, 2, vdupq_n_u64, vld1q_u64, vst1q_u64,
                    vandq_u64, vorrq_u64, veorq_u64, vaddq_u64, vsubq_u64, SHL1_64, vtstq_u64)

/* K independent register groups per query scan. One Myers step depends on the
 * previous one, so a single group is latency-bound; K groups give the core K
 * independent dependency chains to overlap. Unused lanes/groups compute
 * garbage that is never written out. */
#define DEFINE_KGROUP_KERNEL(NAME, VT, ET, LANES, K, DUP, LD, ST, AND, ORR, EOR, ADD, SUB, SHL1, TST) \
static void NAME(const uint8_t *buf_q, const int64_t *off_q, int64_t nq,                          \
                 const uint8_t *buf_c, const int64_t *off_c, int64_t nc,                          \
                 const int64_t *idx, int cnt, int32_t *out) {                                      \
    static _Thread_local ET table[K][256][LANES];                                                  \
    ET last[K][LANES], m0[K][LANES];                                                               \
    memset(table, 0, sizeof(table));                                                               \
    for (int g = 0; g < K; g++)                                                                    \
        for (int l = 0; l < LANES; l++) {                                                          \
            int s = g * LANES + l;                                                                 \
            last[g][l] = 0; m0[g][l] = 0;                                                          \
            if (s >= cnt) continue;                                                                \
            const uint8_t *c = buf_c + off_c[idx[s]];                                              \
            int m = (int)(off_c[idx[s] + 1] - off_c[idx[s]]);                                      \
            for (int i = 0; i < m; i++) table[g][c[i]][l] |= (ET)((ET)1 << i);                     \
            last[g][l] = (ET)((ET)1 << (m - 1));                                                   \
            m0[g][l] = (ET)m;                                                                      \
        }                                                                                          \
    const VT ones = DUP((ET)~(ET)0), one = DUP((ET)1);                                             \
    VT L[K], M0[K];                                                                                \
    for (int g = 0; g < K; g++) { L[g] = LD(last[g]); M0[g] = LD(m0[g]); }                         \
    ET res[LANES];                                                                                 \
    for (int64_t q = 0; q < nq; q++) {                                                             \
        const uint8_t *t = buf_q + off_q[q];                                                       \
        int64_t n = off_q[q + 1] - off_q[q];                                                       \
        VT Pv[K], Mv[K], score[K];                                                                 \
        for (int g = 0; g < K; g++) { Pv[g] = ones; Mv[g] = DUP((ET)0); score[g] = M0[g]; }        \
        for (int64_t j = 0; j < n; j++) {                                                          \
            const uint8_t c = t[j];                                                                \
            for (int g = 0; g < K; g++) {                                                          \
                VT Eq = LD(table[g][c]);                                                           \
                VT Xv = ORR(Eq, Mv[g]);                                                            \
                VT Xh = ORR(EOR(ADD(AND(Eq, Pv[g]), Pv[g]), Pv[g]), Eq);                           \
                VT Ph = ORR(Mv[g], EOR(ORR(Xh, Pv[g]), ones));                                     \
                VT Mh = AND(Pv[g], Xh);                                                            \
                score[g] = ADD(SUB(score[g], TST(Ph, L[g])), TST(Mh, L[g]));                       \
                Ph = ORR(SHL1(Ph), one);                                                           \
                Mh = SHL1(Mh);                                                                     \
                Pv[g] = ORR(Mh, EOR(ORR(Xv, Ph), ones));                                           \
                Mv[g] = AND(Ph, Xv);                                                               \
            }                                                                                      \
        }                                                                                          \
        for (int g = 0; g < K; g++) {                                                              \
            ST(res, score[g]);                                                                     \
            for (int l = 0; l < LANES; l++) {                                                      \
                int s = g * LANES + l;                                                             \
                if (s >= cnt) break;                                                               \
                out[q * nc + idx[s]] = (m0[g][l] == 0) ? (int32_t)n : (int32_t)res[l];             \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
}

#define KINST(W, VT, ET, LANES, K) \
    DEFINE_KGROUP_KERNEL(group##W##_k##K, VT, ET, LANES, K, vdupq_n_u##W, vld1q_u##W, vst1q_u##W, \
                         vandq_u##W, vorrq_u##W, veorq_u##W, vaddq_u##W, vsubq_u##W, SHL1_##W, vtstq_u##W)
KINST(16, uint16x8_t, uint16_t, 8, 2)
KINST(16, uint16x8_t, uint16_t, 8, 4)
KINST(32, uint32x4_t, uint32_t, 4, 2)
KINST(32, uint32x4_t, uint32_t, 4, 4)
KINST(64, uint64x2_t, uint64_t, 2, 2)
KINST(64, uint64x2_t, uint64_t, 2, 4)
#endif

/* Same contract as ed_cross; k = number of interleaved register groups
 * (1, 2 or 4). Returns 0, -1 on allocation failure, -2 for an invalid k. */
int ed_cross_simd_k(const uint8_t *buf_q, const int64_t *off_q, int64_t nq,
                    const uint8_t *buf_c, const int64_t *off_c, int64_t nc,
                    int32_t *out, int k) {
    if (k != 1 && k != 2 && k != 4) return -2;
#if !defined(__ARM_NEON)
    (void)k;
    return ed_cross(buf_q, off_q, nq, buf_c, off_c, nc, out);
#else
    int64_t maxq = 0;
    for (int64_t q = 0; q < nq; q++) {
        int64_t l = off_q[q + 1] - off_q[q];
        if (l > maxq) maxq = l;
    }
    /* Score lanes must hold max(m, n); 16-bit lanes need n < 65536. */
    const int allow16 = maxq < 65535;

    int64_t *cls[4];
    int64_t cnt[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; k++) {
        cls[k] = malloc(sizeof(int64_t) * (nc ? nc : 1));
        if (!cls[k]) { for (int z = 0; z < k; z++) free(cls[z]); return -1; }
    }
    for (int64_t j = 0; j < nc; j++) {
        int64_t m = off_c[j + 1] - off_c[j];
        int k = (m <= 16 && allow16) ? 0 : (m <= 32 ? 1 : (m <= 64 ? 2 : 3));
        cls[k][cnt[k]++] = j;
    }
#define RUN_CLASS(CLS, LANES, F1, F2, F4)                                                     \
    for (int64_t g = 0; g < cnt[CLS];) {                                                          \
        int64_t left = cnt[CLS] - g;                                                              \
        int kk = (k >= 4 && left > 2 * LANES) ? 4 : (k >= 2 && left > LANES) ? 2 : 1;             \
        int take = (int)(left < (int64_t)kk * LANES ? left : (int64_t)kk * LANES);                \
        if (kk == 4) F4(buf_q, off_q, nq, buf_c, off_c, nc, cls[CLS] + g, take, out);             \
        else if (kk == 2) F2(buf_q, off_q, nq, buf_c, off_c, nc, cls[CLS] + g, take, out);        \
        else F1(buf_q, off_q, nq, buf_c, off_c, nc, cls[CLS] + g, take, out);                     \
        g += take;                                                                                \
    }
    RUN_CLASS(0, 8, group16, group16_k2, group16_k4)
    RUN_CLASS(1, 4, group32, group32_k2, group32_k4)
    RUN_CLASS(2, 2, group64, group64_k2, group64_k4)
#undef RUN_CLASS

    int rc = 0;
    if (cnt[3]) {   /* long candidates: scalar, candidate as pattern, loaded once */
        int64_t mx = 0;
        for (int64_t i = 0; i < cnt[3]; i++) {
            int64_t m = off_c[cls[3][i] + 1] - off_c[cls[3][i]];
            if (m > mx) mx = m;
        }
        ws_t ws;
        word *Pv = NULL, *Mv = NULL;
        if (ws_init(&ws, (int)((mx + 63) >> 6)) ||
            !(Pv = malloc(sizeof(word) * ws.W)) || !(Mv = malloc(sizeof(word) * ws.W))) {
            rc = -1;
        } else {
            for (int64_t i = 0; i < cnt[3]; i++) {
                int64_t j = cls[3][i];
                const uint8_t *c = buf_c + off_c[j];
                int m = (int)(off_c[j + 1] - off_c[j]);
                load_pattern(&ws, c, m);
                for (int64_t q = 0; q < nq; q++)
                    out[q * nc + j] = run_text(&ws, m, buf_q + off_q[q],
                                               (int)(off_q[q + 1] - off_q[q]), Pv, Mv);
                clear_pattern(&ws, c, m);
            }
        }
        free(Pv); free(Mv); if (ws.peq) ws_free(&ws);
    }
    for (int k = 0; k < 4; k++) free(cls[k]);
    return rc;
#endif
}

int ed_cross_simd(const uint8_t *buf_q, const int64_t *off_q, int64_t nq,
                  const uint8_t *buf_c, const int64_t *off_c, int64_t nc,
                  int32_t *out) {
    return ed_cross_simd_k(buf_q, off_q, nq, buf_c, off_c, nc, out, 1);
}

/* ------------------------------------------------------------------------
 * Phase 4 step 2: inter-pair SIMD for INDEPENDENT pairs.
 *
 * Every lane has its own pattern and its own text, so each step gathers one
 * Peq entry per lane (scalar loads; NEON has no gather) and a per-lane
 * "active" mask freezes the score of lanes whose text has ended. Pairs are
 * trimmed, the shorter string becomes the pattern, and pairs within a
 * length class are sorted by text length so lanes of a group end together.
 * ---------------------------------------------------------------------- */

typedef struct { const uint8_t *p, *t; int lp, lt; int64_t idx; } pair_t;

static int cmp_lt_desc(const void *x, const void *y) {
    int a = ((const pair_t *)x)->lt, b = ((const pair_t *)y)->lt;
    return (a < b) - (a > b);
}

#if defined(__ARM_NEON)
static const uint8_t ZERO_BYTE = 0;

#define DEFINE_PAIRS_KERNEL(NAME, VT, ET, LANES, K, DUP, LD, ST, AND, ORR, EOR, ADD, SUB, SHL1, TST, CLT) \
static void NAME(const pair_t *pr, int cnt, int32_t *out) {                                       \
    static _Thread_local ET table[K][256][LANES];   /* zero; cleared after each use */            \
    ET last[K][LANES], m0[K][LANES], nlen[K][LANES];                                               \
    const uint8_t *tp[K][LANES];                                                                   \
    int maxn = 0;                                                                                  \
    for (int s = 0; s < K * LANES; s++) {                                                          \
        int g = s / LANES, l = s % LANES;                                                          \
        if (s < cnt) {                                                                             \
            const pair_t *x = &pr[s];                                                              \
            for (int i = 0; i < x->lp; i++) table[g][x->p[i]][l] |= (ET)((ET)1 << i);             \
            last[g][l] = (ET)((ET)1 << (x->lp - 1));                                               \
            m0[g][l] = (ET)x->lp; nlen[g][l] = (ET)x->lt; tp[g][l] = x->t;                         \
            if (x->lt > maxn) maxn = x->lt;                                                        \
        } else {                                                                                   \
            last[g][l] = 0; m0[g][l] = 0; nlen[g][l] = 0; tp[g][l] = &ZERO_BYTE;                   \
        }                                                                                          \
    }                                                                                              \
    const VT ones = DUP((ET)~(ET)0), one = DUP((ET)1);                                             \
    VT L[K], NL[K], Pv[K], Mv[K], score[K];                                                        \
    for (int g = 0; g < K; g++) {                                                                  \
        L[g] = LD(last[g]); NL[g] = LD(nlen[g]); score[g] = LD(m0[g]);                             \
        Pv[g] = ones; Mv[g] = DUP((ET)0);                                                          \
    }                                                                                              \
    ET e[LANES];                                                                                   \
    for (int j = 0; j < maxn; j++) {                                                               \
        const VT J = DUP((ET)j);                                                                   \
        for (int g = 0; g < K; g++) {                                                              \
            for (int l = 0; l < LANES; l++)                                                        \
                e[l] = table[g][(j < (int)nlen[g][l]) ? tp[g][l][j] : 0][l];                       \
            VT Eq = LD(e);                                                                         \
            VT act = CLT(J, NL[g]);                                                                \
            VT Xv = ORR(Eq, Mv[g]);                                                                \
            VT Xh = ORR(EOR(ADD(AND(Eq, Pv[g]), Pv[g]), Pv[g]), Eq);                               \
            VT Ph = ORR(Mv[g], EOR(ORR(Xh, Pv[g]), ones));                                         \
            VT Mh = AND(Pv[g], Xh);                                                                \
            score[g] = ADD(SUB(score[g], AND(TST(Ph, L[g]), act)), AND(TST(Mh, L[g]), act));       \
            Ph = ORR(SHL1(Ph), one);                                                               \
            Mh = SHL1(Mh);                                                                         \
            Pv[g] = ORR(Mh, EOR(ORR(Xv, Ph), ones));                                               \
            Mv[g] = AND(Ph, Xv);                                                                   \
        }                                                                                          \
    }                                                                                              \
    ET res[LANES];                                                                                 \
    for (int g = 0; g < K; g++) {                                                                  \
        ST(res, score[g]);                                                                         \
        for (int l = 0; l < LANES; l++) {                                                          \
            int s = g * LANES + l;                                                                 \
            if (s >= cnt) break;                                                                   \
            out[pr[s].idx] = (int32_t)res[l];                                                      \
            for (int i = 0; i < pr[s].lp; i++) table[g][pr[s].p[i]][l] = 0;                        \
        }                                                                                          \
    }                                                                                              \
}

#define PINST(W, VT, ET, LANES, K) \
    DEFINE_PAIRS_KERNEL(pairs##W##_k##K, VT, ET, LANES, K, vdupq_n_u##W, vld1q_u##W, vst1q_u##W, \
                        vandq_u##W, vorrq_u##W, veorq_u##W, vaddq_u##W, vsubq_u##W, SHL1_##W,    \
                        vtstq_u##W, vcltq_u##W)
PINST(16, uint16x8_t, uint16_t, 8, 1)
PINST(16, uint16x8_t, uint16_t, 8, 2)
PINST(16, uint16x8_t, uint16_t, 8, 4)
PINST(32, uint32x4_t, uint32_t, 4, 1)
PINST(32, uint32x4_t, uint32_t, 4, 2)
PINST(32, uint32x4_t, uint32_t, 4, 4)
PINST(64, uint64x2_t, uint64_t, 2, 1)
PINST(64, uint64x2_t, uint64_t, 2, 2)
PINST(64, uint64x2_t, uint64_t, 2, 4)
#endif

/* Core over pointer/length arrays (shared by the ctypes and CPython paths).
 * Returns 0, -1 on allocation failure, -2 for an invalid k. */
static int pairs_simd_core(const uint8_t *const *pa, const int *la,
                           const uint8_t *const *pb, const int *lb,
                           int64_t n, int32_t *out, int trim, int k) {
    if (k != 1 && k != 2 && k != 4) return -2;
    pair_t *cls[4] = {0};
    int64_t cnt[4] = {0, 0, 0, 0};
    for (int c = 0; c < 4; c++)
        if (!(cls[c] = malloc(sizeof(pair_t) * (n ? n : 1)))) {
            for (int z = 0; z < c; z++) free(cls[z]);
            return -1;
        }
    for (int64_t i = 0; i < n; i++) {
        const uint8_t *a = pa[i], *b = pb[i];
        int x = la[i], y = lb[i];
        if (trim) trim_affix(&a, &x, &b, &y);
        if (x > y) { const uint8_t *t = a; a = b; b = t; int z = x; x = y; y = z; }
        if (x == 0) { out[i] = y; continue; }
#if defined(__ARM_NEON)
        int c = (x <= 16 && y < 65535) ? 0 : (x <= 32 ? 1 : (x <= 64 ? 2 : 3));
#else
        int c = 3;
#endif
        cls[c][cnt[c]++] = (pair_t){a, b, x, y, i};
    }
    int rc = 0;
#if defined(__ARM_NEON)
    const int lanes[3] = {8, 4, 2};
    for (int c = 0; c < 3; c++) {
        qsort(cls[c], (size_t)cnt[c], sizeof(pair_t), cmp_lt_desc);
        for (int64_t g = 0; g < cnt[c];) {
            int64_t left = cnt[c] - g;
            int kk = (k >= 4 && left > 2 * lanes[c]) ? 4 : (k >= 2 && left > lanes[c]) ? 2 : 1;
            int take = (int)(left < (int64_t)kk * lanes[c] ? left : (int64_t)kk * lanes[c]);
            const pair_t *p = cls[c] + g;
            if (c == 0) { if (kk == 4) pairs16_k4(p, take, out); else if (kk == 2) pairs16_k2(p, take, out); else pairs16_k1(p, take, out); }
            if (c == 1) { if (kk == 4) pairs32_k4(p, take, out); else if (kk == 2) pairs32_k2(p, take, out); else pairs32_k1(p, take, out); }
            if (c == 2) { if (kk == 4) pairs64_k4(p, take, out); else if (kk == 2) pairs64_k2(p, take, out); else pairs64_k1(p, take, out); }
            g += take;
        }
    }
#endif
    if (cnt[3]) {   /* long patterns: scalar */
        int mx = 0;
        for (int64_t i = 0; i < cnt[3]; i++) if (cls[3][i].lp > mx) mx = cls[3][i].lp;
        ws_t ws = {0};
        word *Pv = NULL, *Mv = NULL;
        if (ws_init(&ws, (mx + 63) >> 6) || !(Pv = malloc(sizeof(word) * ws.W)) ||
            !(Mv = malloc(sizeof(word) * ws.W))) {
            rc = -1;
        } else {
            for (int64_t i = 0; i < cnt[3]; i++) {
                const pair_t *x = &cls[3][i];
                out[x->idx] = pair_distance(&ws, Pv, Mv, x->p, x->lp, x->t, x->lt, 0);
            }
        }
        free(Pv); free(Mv); if (ws.peq) ws_free(&ws);
    }
    for (int c = 0; c < 4; c++) free(cls[c]);
    return rc;
}

int ed_pairs_simd(const uint8_t *buf_a, const int64_t *off_a,
                  const uint8_t *buf_b, const int64_t *off_b,
                  int64_t n, int32_t *out, int trim, int k) {
    const uint8_t **pa = malloc(sizeof(*pa) * (n ? n : 1)), **pb = malloc(sizeof(*pb) * (n ? n : 1));
    int *la = malloc(sizeof(int) * (n ? n : 1)), *lb = malloc(sizeof(int) * (n ? n : 1));
    int rc = -1;
    if (pa && pb && la && lb) {
        for (int64_t i = 0; i < n; i++) {
            pa[i] = buf_a + off_a[i]; la[i] = (int)(off_a[i + 1] - off_a[i]);
            pb[i] = buf_b + off_b[i]; lb[i] = (int)(off_b[i + 1] - off_b[i]);
        }
        rc = pairs_simd_core(pa, la, pb, lb, n, out, trim, k);
    }
    free(pa); free(pb); free(la); free(lb);
    return rc;
}
