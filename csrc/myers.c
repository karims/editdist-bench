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
