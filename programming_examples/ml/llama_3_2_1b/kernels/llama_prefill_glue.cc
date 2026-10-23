//===- llama_prefill_glue.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The prefill's per-row glue beside the decode's own kernels: a row's qkv
// into the attention's chunks and the KV cache's rows (the decode's
// llama_gemv_array_qkv_comb_cs and llama_comb_rope_rows, per KV group), and
// the fp32 residual add.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"
#include "llama_kv_row.h"

static constexpr int kHD = 64;
static constexpr int kGroups = 8;
static constexpr int kGroupFp = 4 * kHD + 2 * kHD;
static constexpr int kQChunk = 4 * kHD + 4 * 8;
static constexpr int kRowBytes = 2 * kHD + 8;
static constexpr int kChunkStride = 448;
static constexpr int kCols = 2048;

extern "C" {

// Row `fp` of the GEMM array's qkv (per KV group [q 256 | k 64 | v 64]
// fp32) at the position whose RoPE cos/sin is `cs` (bf16 cos[64], sin[64]):
// per group, chunks[g] = [the roped, requantized Q heads 288 B | the KV row
// 136 B | 0] and rows[g] = the KV row ([k int8 | v int8 | k_scale |
// v_scale]).
void llama_prefill_qkv_row(const float *restrict fp, const int8_t *restrict cs,
                           int8_t *restrict chunks, int8_t *restrict rows) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int g = 0; g < kGroups; g++) {
    const float *q = fp + g * kGroupFp;
    int8_t *chunk = chunks + g * kChunkStride;
    rope_q_requant4(q, reinterpret_cast<const uint16_t *>(cs), chunk);
    alignas(64) int8_t kvfp[2 * kHD * 4 + 2 * kHD * 2];
    copy_words<2 * kHD * 4>(kvfp, q + 4 * kHD);
    copy_words<2 * kHD * 2>(kvfp + 2 * kHD * 4, cs);
    alignas(64) int8_t row[kRowBytes + 56];
    llama_kv_row_quant<kHD>(kvfp, row);
    copy_words<kRowBytes>(chunk + kQChunk, row);
    zero_words<kChunkStride - kQChunk - kRowBytes>(chunk + kQChunk + kRowBytes);
    copy_words<kRowBytes>(rows + g * kRowBytes, row);
  }
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// out = x + proj in fp32, x the residual row, proj a projection's row.
void llama_prefill_add(const float *restrict x, const float *restrict proj,
                       float *restrict out) {
  event0();
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < kCols / 16; i++)
    ::aie::store_v(out + i * 16, ::aie::add(::aie::load_v<16>(x + i * 16),
                                            ::aie::load_v<16>(proj + i * 16)));
  ::aie::set_rounding(prev_rnd);
  event1();
}

} // extern "C"
