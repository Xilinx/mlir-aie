//===- llama_gemm_int8_srs_tiled_attn.cc ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Attention-half integration variant of the tiled decode GEMV: one .o
// exposing two K-shapes for the attention path:
//   - K=2048: q_proj (h1[D=2048] -> qf[N=64] for single-head)
//   - K=64:   o_proj (attn[HD=64] -> op[N=D=2048])
//
// Same template logic as `llama_gemm_int8_srs_tiled_ffn.cc` and the
// standalone `llama_gemm_int8_srs_tiled.cc`. They're separate .o files
// (and only one is linked into any given xclbin) because aiecc would
// hit duplicate-symbol errors otherwise. When 6c.3b.3 merges FFN + attn
// into one single-layer xclbin, we'll consolidate into a single kernel
// file with all four (K, N_TILE) entries.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr int32_t I8_MAX =  127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

template <int kK, int kNTile>
static inline void gemm_tile_impl(int8_t *restrict act,
                                  int8_t *restrict w_tile,
                                  int8_t *restrict out_tile,
                                  int32_t right_shift) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t  *weights = w_tile;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }

    int32_t sum =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    out_tile[n] = (int8_t)s;
  }
}

extern "C" {

// q_proj: K = D = 2048, N_TILE = 4.
// Worker passes the FULL output buffer + a tile_idx scalar; kernel
// pointer-offsets to the right slice.
void llama_gemm_tiled_K2048_N4(int8_t *restrict act,
                               int8_t *restrict w_tile,
                               int8_t *restrict out_full,
                               int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// o_proj: K = HEAD_DIM = 64, N_TILE = 4. K=64 fits in one MAC group so
// the inner loop trivially unrolls.
void llama_gemm_tiled_K64_N4(int8_t *restrict act,
                             int8_t *restrict w_tile,
                             int8_t *restrict out_full,
                             int32_t tile_idx,
                             int32_t right_shift) {
  event0();
  gemm_tile_impl<64, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

} // extern "C"
