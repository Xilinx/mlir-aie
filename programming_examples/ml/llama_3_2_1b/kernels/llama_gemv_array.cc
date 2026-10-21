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

#ifndef LLAMA_GEMV_ARRAY_KSET
#define LLAMA_GEMV_ARRAY_KSET 8
#endif

// Residual units (o, down): each core owns 256 of the 2048 rows and emits a
// kResidBlock-byte block [proj fp32 x256 | pad], 64-byte aligned for the
// vector stores.
static constexpr int kResidRows = 256;
static constexpr int kResidBlock = kResidRows * 4 + 64;
// A residual row: the norm's [int8 x | scale | pad], then x in fp32 here.
static constexpr int kFpAt = 2048 + 64;

extern "C" {

void llama_gemv_array_qkv_comb(int8_t *restrict fp, int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  q_requant_heads<4, 64>(reinterpret_cast<const float *>(fp), out, out + 256);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  copy_words<512>(out + 288, fp + 1024);
  zero_words<256>(out + 800);
  event1();
}

// llama_gemv_array_qkv_comb for activation `row` of fp ([row][q 256 | k 64 |
// v 64] fp32) with its position's RoPE cos/sin, which ride the head of the
// core's weight lane each layer (cs: 256 B a position), in the cs region.
void llama_gemv_array_qkv_comb_cs(int8_t *restrict fp, int8_t *restrict cs,
                                  int8_t *restrict out, int32_t row) {
  llama_gemv_array_qkv_comb(fp + row * 384 * 4, out);
  copy_words<256>(out + 800, cs + row * 256);
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

// The glue core's residual add over the joined blocks (lo: cores 0-3, then
// the fp32[2048] residual; hi: cores 4-7): x = resid + proj in fp32, at
// out + kFpAt, and x requantized for the next norm at out ([int8[2048] |
// scale | 0]).
void llama_gemv_array_residual_add(int8_t *restrict lo, int8_t *restrict hi,
                                   int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *r = reinterpret_cast<const float *>(lo + 4 * kResidBlock);
  float *x = reinterpret_cast<float *>(out + kFpAt);
  ::aie::vector<int32, 16> mag = ::aie::zeros<int32, 16>();
  for (int c = 0; c < 8; c++) {
    const float *proj = reinterpret_cast<const float *>((c < 4 ? lo : hi) +
                                                        (c % 4) * kResidBlock);
    for (int i = 0; i < kResidRows / 16; i++) {
      ::aie::vector<float, 16> xf =
          ::aie::add(::aie::load_v<16>(r + c * kResidRows + i * 16),
                     ::aie::load_v<16>(proj + i * 16));
      ::aie::store_v(x + c * kResidRows + i * 16, xf);
      mag = ::aie::max(
          mag, ::aie::bit_and(int32(0x7fffffff), xf.template cast_to<int32>()));
    }
  }
  int32_t absmax_bits = ::aie::reduce_max(mag);
  if (absmax_bits < 0x2b8cbccc)
    absmax_bits = 0x2b8cbccc;
  float x_scale = __builtin_bit_cast(float, absmax_bits) * (1.0f / 127.0f);
  ::aie::vector<float, 16> inv_v =
      ::aie::broadcast<float, 16>(recip_rne(x_scale));
  for (int i = 0; i < 8 * kResidRows / 16; i++)
    ::aie::store_v(out + i * 16, vec_round_to_i8(vec_mul_rne(
                                     ::aie::load_v<16>(x + i * 16), inv_v)));
  *reinterpret_cast<float *>(out + 2048) = x_scale;
  zero_words<kFpAt - 2052>(out + 2052);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

} // extern "C"
