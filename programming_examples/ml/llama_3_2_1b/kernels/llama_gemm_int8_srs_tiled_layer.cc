//===- llama_gemm_int8_srs_tiled_layer.cc -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Full-single-layer integration variant of the tiled decode GEMV: one
// .o exposing every K-shape a single Llama 3.2 1B decoder layer at
// D=2048, single-head, HD=8192 needs:
//
//   K=2048  — q_proj (D->HEAD_DIM), gate_proj/up_proj (D->HD)
//   K=64    — o_proj (HEAD_DIM->D)        [single-head only; 6c.3b.2/3]
//   K=8192  — down_proj (HD->D)
//
// Three sister files exist for narrower configurations
// (`..._ffn.cc`, `..._attn.cc`, standalone `..._tiled.cc`) so each
// integration xclbin can link only the kernel symbols it needs without
// hitting duplicate-symbol errors. This file is the "everything for one
// decoder layer" superset.
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

// q_proj (single-head: K=D=2048, output N=QD=64).
// Same template as gate/up, distinct symbol so IRON's Kernel registry
// can carry the QD-typed output type instead of the HD-typed one.
void llama_gemm_tiled_K2048_N4_qproj(int8_t *restrict act,
                                     int8_t *restrict w_tile,
                                     int8_t *restrict out_full,
                                     int32_t tile_idx,
                                     int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// gate_proj + up_proj (K=D=2048, output N=HD=8192).
void llama_gemm_tiled_K2048_N4_hproj(int8_t *restrict act,
                                     int8_t *restrict w_tile,
                                     int8_t *restrict out_full,
                                     int32_t tile_idx,
                                     int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// o_proj (single-head: K=HEAD_DIM=64 -> D=2048).
void llama_gemm_tiled_K64_N4(int8_t *restrict act,
                             int8_t *restrict w_tile,
                             int8_t *restrict out_full,
                             int32_t tile_idx,
                             int32_t right_shift) {
  event0();
  gemm_tile_impl<64, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// down_proj (K=HD=8192 -> D=2048).
void llama_gemm_tiled_K8192_N4(int8_t *restrict act,
                               int8_t *restrict w_tile,
                               int8_t *restrict out_full,
                               int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<8192, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

} // extern "C"
