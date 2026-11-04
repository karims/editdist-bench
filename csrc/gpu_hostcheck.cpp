/* Host build of the GPU device functions, so their logic is tested on the CPU
 * (tests/test_gpu_logic.py) without a GPU. */
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "gpu_myers.cuh"

extern "C" int gpu_pair_distance_host(const u8 *a, int la, const u8 *b, int lb, int trim) {
    switch (pair_class(la, lb)) {
    case 0: return pair_distance_swar<16>(a, la, b, lb, trim);
    case 1: return pair_distance_swar<32>(a, la, b, lb, trim);
    case 2: return pair_distance_swar<64>(a, la, b, lb, trim);
    case 3: return pair_distance_swar<128>(a, la, b, lb, trim);
    case 4: return pair_distance_swar<256>(a, la, b, lb, trim);
    default: return -1;
    }
}

/* Same as the cross kernel's per-thread work: pattern a via a Peq table. */
extern "C" int gpu_peq_distance_host(const u8 *a, int la, const u8 *b, int lb) {
    if (la == 0) return lb;
    int nw = (la + 63) / 64;
    int NW = nw <= 1 ? 1 : nw <= 2 ? 2 : nw <= 4 ? 4 : nw <= 8 ? 8 : 0;
    if (!NW) return -1;
    std::vector<u64> peq((std::size_t)256 * NW, 0);
    for (int i = 0; i < la; i++) peq[(std::size_t)a[i] * NW + i / 64] |= 1ull << (i % 64);
    switch (NW) {
    case 1: return text_distance_peq<1>(peq.data(), la, b, lb);
    case 2: return text_distance_peq<2>(peq.data(), la, b, lb);
    case 4: return text_distance_peq<4>(peq.data(), la, b, lb);
    default: return text_distance_peq<8>(peq.data(), la, b, lb);
    }
}
