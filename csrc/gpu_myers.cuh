/* Bit-parallel edit distance (Myers 1999 / Hyyro) for the GPU phase.
 *
 * Everything here compiles both as CUDA device code and as ordinary C++, so
 * the exact same functions are unit-tested on the CPU (csrc/gpu_hostcheck.cpp)
 * before they ever run on a GPU.
 *
 * Two ways to get Eq (the bitmask of pattern positions equal to a text char):
 *   - pairs: every thread has its own pattern, so there is no shared lookup
 *     table. The pattern lives in registers as packed bytes and Eq is computed
 *     by comparing 4 bytes at a time (SWAR), with no memory traffic.
 *   - cross: one pattern (the query) per thread block, so its 256-row lookup
 *     table (Peq) is built once in shared memory and every thread reads it.
 */
#pragma once

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

/* Minimal std::conditional (NVRTC has no standard library headers). */
template <bool B, typename A, typename C> struct pick { typedef A type; };
template <typename A, typename C> struct pick<false, A, C> { typedef C type; };

/* Bits 0..3 = which of the 4 bytes of w equal the byte replicated in c4. */
HD u32 eq4(u32 w, u32 c4) {
#if defined(__CUDA_ARCH__)
    u32 y = (__vcmpeq4(w, c4) & 0x80808080u) >> 7;            /* 1 per equal byte */
#else
    u32 x = w ^ c4;                                            /* zero byte <=> equal */
    u32 t = ~((((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) | 0x7f7f7f7fu);   /* exact */
    u32 y = t >> 7;
#endif
    return ((y * 0x00204081u) >> 21) & 0xfu;                   /* gather bits 0,8,16,24 */
}

/* One Myers block step on word type T. hin in {-1,0,+1}; returns the
 * horizontal delta leaving bit `outbit`. Bits above the pattern length are
 * don't-care: carries and shifts only move information upwards. */
template <typename T>
HD int block_step(T &Pv, T &Mv, T Eq, int hin, int outbit) {
    T hneg = (T)(hin < 0);
    T Xv = Eq | Mv;
    Eq |= hneg;
    T Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;
    T Ph = Mv | ~(Xh | Pv);
    T Mh = Pv & Xh;
    int hout = (int)((Ph >> outbit) & 1) - (int)((Mh >> outbit) & 1);
    Ph = (Ph << 1) | (T)(hin > 0);
    Mh = (Mh << 1) | hneg;
    Pv = Mh | ~(Xv | Ph);
    Mv = Ph & Xv;
    return hout;
}

/* Exact: d(xPy, xQy) == d(P, Q). */
HD void trim_affix(const u8 *&a, int &la, const u8 *&b, int &lb) {
    int n = la < lb ? la : lb, p = 0;
    while (p < n && a[p] == b[p]) p++;
    int s = 0;
    while (s < n - p && a[la - 1 - s] == b[lb - 1 - s]) s++;
    a += p; b += p; la -= p + s; lb -= p + s;
}

/* Distance of one pair whose SHORTER string has length <= CAP (CAP in
 * 16, 32, 64, 128, 256). The pattern is held in registers: all loops over it
 * are fully unrolled with compile-time indices, so nothing spills to memory. */
template <int CAP>
HD int pair_distance_swar(const u8 *a, int la, const u8 *b, int lb, int trim) {
    typedef typename pick<(CAP <= 32), u32, u64>::type T;
    constexpr int BITS = 8 * (int)sizeof(T);
    constexpr int NW = (CAP + BITS - 1) / BITS;      /* Myers words */
    constexpr int NP = CAP / 4;                      /* packed pattern u32s */
    if (trim) trim_affix(a, la, b, lb);
    if (la > lb) { const u8 *t = a; a = b; b = t; int x = la; la = lb; lb = x; }
    if (la == 0) return lb;

    u32 pw[NP];
#pragma unroll
    for (int k = 0; k < NP; k++) {
        u32 w = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int pos = 4 * k + i;
            w |= (u32)(pos < la ? a[pos] : 0) << (8 * i);
        }
        pw[k] = w;
    }
    T Pv[NW], Mv[NW];
#pragma unroll
    for (int w = 0; w < NW; w++) { Pv[w] = ~(T)0; Mv[w] = 0; }
    const int nw = (la + BITS - 1) / BITS;           /* words actually used */
    const int lastbit = (la - 1) % BITS;
    int score = la;
    for (int j = 0; j < lb; j++) {
        const u32 c4 = (u32)b[j] * 0x01010101u;
        int h = 1;                                   /* top row: +1 per column */
#pragma unroll
        for (int w = 0; w < NW; w++) {
            if (w < nw) {
                T Eq = 0;
#pragma unroll
                for (int k = 0; k < BITS / 4; k++)
                    if (w * BITS + 4 * k < CAP)      /* compile-time: skip unused positions */
                        Eq |= (T)eq4(pw[w * (BITS / 4) + k], c4) << (4 * k);
                if (w < nw - 1) h = block_step<T>(Pv[w], Mv[w], Eq, h, BITS - 1);
                else score += block_step<T>(Pv[w], Mv[w], Eq, h, lastbit);
            }
        }
    }
    return score;
}

/* Distance between a pattern given by its Peq table (row c = 256 entries of
 * NW u64 words, laid out peq[c * NW + w]) and a text. Pattern length m >= 1,
 * m <= 64 * NW. Used by the cross kernel with peq in shared memory. */
template <int NW>
HD int text_distance_peq(const u64 *peq, int m, const u8 *t, int n) {
    u64 Pv[NW], Mv[NW];
#pragma unroll
    for (int w = 0; w < NW; w++) { Pv[w] = ~0ull; Mv[w] = 0; }
    const int nw = (m + 63) / 64, lastbit = (m - 1) % 64;
    int score = m;
    for (int j = 0; j < n; j++) {
        const u64 *row = peq + (int)t[j] * NW;
        int h = 1;
#pragma unroll
        for (int w = 0; w < NW; w++) {
            if (w < nw) {
                if (w < nw - 1) h = block_step<u64>(Pv[w], Mv[w], row[w], h, 63);
                else score += block_step<u64>(Pv[w], Mv[w], row[w], h, lastbit);
            }
        }
    }
    return score;
}

/* Capacity class for a pair: smallest CAP >= the shorter length, or -1 if
 * longer than 256 (handled by the CPU). Classes 0..4 = 16,32,64,128,256. */
HD int pair_class(int la, int lb) {
    int m = la < lb ? la : lb;
    return m <= 16 ? 0 : m <= 32 ? 1 : m <= 64 ? 2 : m <= 128 ? 3 : m <= 256 ? 4 : -1;
}
