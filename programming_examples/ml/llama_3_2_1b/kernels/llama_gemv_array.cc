//===- llama_gemv_array.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array cores' glue beside the projections (llama_gemv_mma.cc): the
// fused qkv combine, the residual add's partials and their finalize, and the
// lm_head's top-k set packing.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

// Residual units (o, down): each core owns 256 of the 2048 rows and emits a
// kResidBlock-byte block [proj fp32 x256 | pad], 64-byte aligned for the
// vector stores.
static constexpr int kResidRows = 256;
static constexpr int kResidBlock = kResidRows * 4 + 64;

extern "C" {

// The fused qkv combine for activation `row` of fp ([row][q 256 | k 64 | v
// 64] fp32), with its position's RoPE cos/sin, which ride the head of the
// core's weight lane each layer (cs: 256 B a position, bf16 cos[64] then
// sin[64]), in the cs region: the core's 4 Q heads roped in fp32 as
// llama_kv_row.h ropes k, then requantized per head; then k and v in fp32
// and the cos/sin for the KV row's rope.
void llama_gemv_array_qkv_comb_cs(int8_t *restrict fp, int8_t *restrict cs,
                                  int8_t *restrict out, int32_t row) {
  event0();
  constexpr int kHD = 64;
  constexpr int kHalf = kHD / 2;
  const float *q = reinterpret_cast<const float *>(fp + row * 384 * 4);
  const uint16_t *cs_bits = reinterpret_cast<const uint16_t *>(cs + row * 256);
  alignas(64) float cs_f[2 * kHD];
  for (int i = 0; i < 2 * kHD; i++)
    cs_f[i] = __builtin_bit_cast(float, (uint32_t)cs_bits[i] << 16);
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  alignas(64) float roped[4 * kHD];
  for (int h = 0; h < 4; h++)
    for (int j = 0; j < kHalf; j += 16) {
      ::aie::vector<float, 16> x1 = ::aie::load_v<16>(q + h * kHD + j);
      ::aie::vector<float, 16> x2 = ::aie::load_v<16>(q + h * kHD + kHalf + j);
      ::aie::store_v(
          roped + h * kHD + j,
          ::aie::sub(vec_mul_rne(x1, ::aie::load_v<16>(cs_f + j)),
                     vec_mul_rne(x2, ::aie::load_v<16>(cs_f + kHD + j))));
      ::aie::store_v(
          roped + h * kHD + kHalf + j,
          ::aie::add(
              vec_mul_rne(x2, ::aie::load_v<16>(cs_f + kHalf + j)),
              vec_mul_rne(x1, ::aie::load_v<16>(cs_f + kHD + kHalf + j))));
    }
  q_requant_heads<4, kHD>(roped, out, out + 256);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  copy_words<512>(out + 288, q + 256);
  copy_words<256>(out + 800, cs + row * 256);
  event1();
}

// The first `rows` positions' RoPE cos/sin (256 B each) from the head of the
// core's weight lane into cs, before the lane is released.
void llama_gemv_array_take_cs(int8_t *restrict lane, int8_t *restrict cs,
                              int32_t rows) {
  for (int32_t r = 0; r < rows; r++)
    copy_words<256>(cs + r * 256, lane + r * 256);
}

// Core's projection rows for activation `row` (fp32, fp[256 row..256 row +
// 256)) into its block for the glue core's residual add.
void llama_gemv_array_proj_block(int8_t *restrict fp, int8_t *restrict blk,
                                 int32_t row) {
  copy_words<kResidRows * 4>(blk, fp + row * kResidRows * 4);
}

// The glue core's residual add over the joined blocks (lo: cores 0-3; hi:
// cores 4-7) and the fp32[2048] residual: out = resid + proj in fp32, the
// next norm's input and the next unit's residual.
void llama_gemv_array_residual_add(int8_t *restrict lo, int8_t *restrict hi,
                                   int8_t *restrict resid,
                                   int8_t *restrict out) {
  event0();
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *r = reinterpret_cast<const float *>(resid);
  float *x = reinterpret_cast<float *>(out);
  for (int c = 0; c < 8; c++) {
    const float *proj = reinterpret_cast<const float *>((c < 4 ? lo : hi) +
                                                        (c % 4) * kResidBlock);
    for (int i = 0; i < kResidRows / 16; i++)
      ::aie::store_v(x + c * kResidRows + i * 16,
                     ::aie::add(::aie::load_v<16>(r + c * kResidRows + i * 16),
                                ::aie::load_v<16>(proj + i * 16)));
  }
  ::aie::set_rounding(prev_rnd);
  event1();
}

} // extern "C"
