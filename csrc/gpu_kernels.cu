/* CUDA kernels; compiled at run time by CuPy (NVRTC) from bench/cudamyers.py.
 * The per-pair work is in gpu_myers.cuh, which is unit-tested on the CPU. */
#include "gpu_myers.cuh"

/* ---- independent pairs: one thread per pair, pattern in registers ---------- */
template <int CAP>
__device__ __forceinline__ void pairs_body(const u8 *buf_a, const long long *off_a,
                                           const u8 *buf_b, const long long *off_b,
                                           const int *idx, int n, int *out, int trim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i = idx[t];
    long long a0 = off_a[i], b0 = off_b[i];
    out[i] = pair_distance_swar<CAP>(buf_a + a0, (int)(off_a[i + 1] - a0),
                                     buf_b + b0, (int)(off_b[i + 1] - b0), trim);
}

#define PAIRS_KERNEL(CAP)                                                                  \
extern "C" __global__ void __launch_bounds__(256)                                          \
ed_pairs_c##CAP(const u8 *buf_a, const long long *off_a, const u8 *buf_b,                  \
                const long long *off_b, const int *idx, int n, int *out, int trim) {       \
    pairs_body<CAP>(buf_a, off_a, buf_b, off_b, idx, n, out, trim);                         \
}
PAIRS_KERNEL(16)
PAIRS_KERNEL(32)
PAIRS_KERNEL(64)
PAIRS_KERNEL(128)
PAIRS_KERNEL(256)

/* ---- cross: one block per query (the pattern), Peq table in shared memory,
 *      threads stride over candidates (the texts). out is nq x nc, row-major. */
template <int NW>
__device__ __forceinline__ void cross_body(const u8 *buf_q, const long long *off_q,
                                           const int *qidx, const u8 *buf_c,
                                           const long long *off_c, int nc, int *out) {
    extern __shared__ u64 peq[];                    /* 256 * NW words */
    const int q = qidx[blockIdx.x];
    const u8 *p = buf_q + off_q[q];
    const int m = (int)(off_q[q + 1] - off_q[q]);
    for (int k = threadIdx.x; k < 256 * NW; k += blockDim.x) peq[k] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x)
        atomicOr(&peq[(int)p[i] * NW + (i >> 6)], 1ull << (i & 63));
    __syncthreads();
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < nc; c += gridDim.y * blockDim.x) {
        const long long c0 = off_c[c];
        const int n = (int)(off_c[c + 1] - c0);
        out[(long long)q * nc + c] = (m == 0) ? n : text_distance_peq<NW>(peq, m, buf_c + c0, n);
    }
}

#define CROSS_KERNEL(NW)                                                                   \
extern "C" __global__ void __launch_bounds__(256)                                          \
ed_cross_w##NW(const u8 *buf_q, const long long *off_q, const int *qidx,                   \
               const u8 *buf_c, const long long *off_c, int nc, int *out) {                \
    cross_body<NW>(buf_q, off_q, qidx, buf_c, off_c, nc, out);                              \
}
CROSS_KERNEL(1)
CROSS_KERNEL(2)
CROSS_KERNEL(4)
CROSS_KERNEL(8)
