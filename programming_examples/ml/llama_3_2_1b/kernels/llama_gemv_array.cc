//===- llama_gemv_array.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array cores' glue beside the projections (llama_gemv_mma.cc): the
// fused qkv combine, the RoPE row's take, and the residual add.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

// Residual units (o, down): each core owns 256 of the 2048 rows and emits a
// kResidBlock-byte block [proj fp32 x256 | pad], 64-byte aligned for the
// vector stores.
static constexpr int kResidRows = 256;
static constexpr int kResidBlock = kResidRows * 4 + 64;
// A row's fused qkv outputs in fp32 (aie2_gemv_array.QKV_COLS).
static constexpr int kQkvCols = 384;

#ifndef LLAMA_GEMV_M_MAX
#define LLAMA_GEMV_M_MAX 4
#endif

// cs in part (8 partials a row, kResidRows rows): its groups 16-19, which
// only mma_down writes, after the layer's last qkv_comb_cs.
static constexpr int kCsInPart = 4096;
static constexpr int kCsBytes = 256;
static_assert(kCsInPart + LLAMA_GEMV_M_MAX * kCsBytes <= 8 * kResidRows * 4,
              "a pass's cs rows fit in part");

extern "C" {

// The fused qkv combine for activation `row` of fp ([row][q 256 | k 64 | v
// 64] fp32), with its position's RoPE cos/sin, which ride the head of the
// core's weight lane each layer (cs: 256 B a position, bf16 cos[64] then
// sin[64]), in the cs region of `part`: the core's 4 Q heads roped in fp32
// as llama_kv_row.h ropes k, then requantized per head; then k and v in
// fp32 and the cos/sin for the KV row's rope.
void llama_gemv_array_qkv_comb_cs(int8_t *restrict fp, int32_t *restrict part,
                                  int8_t *restrict out, int32_t row) {
  event0();
  int8_t *cs = reinterpret_cast<int8_t *>(part) + kCsInPart;
  const float *q = reinterpret_cast<const float *>(fp + row * kQkvCols * 4);
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  rope_q_requant4(q, reinterpret_cast<const uint16_t *>(cs + row * kCsBytes),
                  out);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  copy_words<512>(out + 288, q + 256);
  copy_words<kCsBytes>(out + 800, cs + row * kCsBytes);
  event1();
}

// The first `rows` positions' RoPE cos/sin (kCsBytes each) from the head of
// the core's weight lane into cs (part's bytes from kCsInPart), before the
// lane is released.
void llama_gemv_array_take_cs(int8_t *restrict lane, int32_t *restrict part,
                              int32_t rows) {
  int8_t *cs = reinterpret_cast<int8_t *>(part) + kCsInPart;
  for (int32_t r = 0; r < rows; r++)
    copy_words<kCsBytes>(cs + r * kCsBytes, lane + r * kCsBytes);
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
