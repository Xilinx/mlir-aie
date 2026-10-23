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

// The position's cos and sin (bf16 cos[64], sin[64]) as x_bf16_mul's
// factors: [cos 0..31, cos 32..63, sin 0..31, sin 32..63].
struct RopeFactors {
  Bf16Factor f[4];
  int lo, hi; // the nonzero exponent fields' range
};

// One KV group through rope_q_requant4 and llama_kv_row_quant's arithmetic
// in fast forms (x_bf16_mul, vec_requant_i8), where x_bf16_mul's products
// stay normal: chunk gets the Q heads and their tails, row the KV row. Out
// of line, as is qkv_group, so that their frames don't add up.
__attribute__((noinline)) static bool
qkv_group_fast(const float *restrict q, const RopeFactors &cs,
               const ::aie::accum<acc64, 32> &init, int8_t *restrict chunk,
               int8_t *restrict row) {
  const int32_t *qb = reinterpret_cast<const int32_t *>(q);
  ExpRange xr = exp_range_empty();
  for (int i = 0; i < 5 * kHD / 32; i++)
    exp_range_add(xr, x_planes(::aie::load_v<32>(qb + 32 * i)).h);
  int x_lo, x_hi;
  exp_range_get(xr, x_lo, x_hi);
  if (x_lo <= x_hi && cs.lo <= cs.hi &&
      !(x_lo + cs.lo - 127 >= 2 && x_hi + cs.hi - 126 <= 253))
    return false;
  // Heads 0-3 Q, 4 K: x1 = head[0:32], x2 = head[32:64];
  // [x1 * cos_lo - x2 * sin_lo | x2 * cos_hi + x1 * sin_hi].
  alignas(64) float roped[5][kHD];
  ::aie::vector<int32, 8> absmax = ::aie::zeros<int32, 8>();
  for (int h = 0; h < 5; h++) {
    ::aie::vector<int32, 32> b1 = ::aie::load_v<32>(qb + h * kHD);
    ::aie::vector<int32, 32> b2 = ::aie::load_v<32>(qb + h * kHD + 32);
    XPlanes x1 = x_planes(b1), x2 = x_planes(b2);
    ::aie::vector<float, 32> r1 =
        ::aie::sub(x_bf16_mul(b1, x1, cs.f[0], init).cast_to<float>(),
                   x_bf16_mul(b2, x2, cs.f[2], init).cast_to<float>());
    ::aie::vector<float, 32> r2 =
        ::aie::add(x_bf16_mul(b2, x2, cs.f[1], init).cast_to<float>(),
                   x_bf16_mul(b1, x1, cs.f[3], init).cast_to<float>());
    ::aie::store_v(roped[h], r1);
    ::aie::store_v(roped[h] + 32, r2);
    absmax.set(::aie::reduce_max(::aie::max(
                   ::aie::bit_and(int32(0x7fffffff), r1.cast_to<int32>()),
                   ::aie::bit_and(int32(0x7fffffff), r2.cast_to<int32>()))),
               h);
  }
  absmax.set(
      ::aie::reduce_max(::aie::max(
          ::aie::bit_and(int32(0x7fffffff), ::aie::load_v<32>(qb + 5 * kHD)),
          ::aie::bit_and(int32(0x7fffffff),
                         ::aie::load_v<32>(qb + 5 * kHD + 32)))),
      5);
  // Every scale max(absmax, 1e-12) * (1 / 127) at once.
  alignas(32) float scale[8];
  ::aie::store_v(
      scale, vec_mul_rne(::aie::max(absmax, int32(0x2b8cbccc)).cast_to<float>(),
                         ::aie::broadcast<float, 8>(1.0f / 127.0f)));
  // Their reciprocals at once, the lanes past them 1.
  alignas(64) float inv_f[32];
  {
    const ::aie::vector<float, 8> s8 = ::aie::load_v<8>(scale);
    ::aie::vector<int32, 32> sb =
        ::aie::broadcast<float, 32>(1.0f).cast_to<int32>();
    ::aie::vector<int32, 32> r0 = sb;
    sb.insert(0, s8.cast_to<int32>());
    r0.insert(0, ::aie::inv(s8).cast_to<int32>());
    ::aie::store_v(inv_f, vec_recip_rne_bits(sb, r0).cast_to<float>());
  }
  for (int h = 0; h < 6; h++) {
    const RequantFactor inv = requant_factor(inv_f[h]);
    const int32_t *src =
        h < 5 ? reinterpret_cast<const int32_t *>(roped[h]) : qb + 5 * kHD;
    int8_t *dst = h < 4 ? chunk + h * kHD : row + (h - 4) * kHD;
    for (int i = 0; i < 2; i++)
      ::aie::store_v(dst + 32 * i,
                     vec_requant_i8(::aie::load_v<32>(src + 32 * i), inv));
  }
  for (int h = 0; h < 4; h++) {
    reinterpret_cast<llama_word *>(chunk + 4 * kHD + h * 8)[0] =
        __builtin_bit_cast(int32_t, scale[h]);
    reinterpret_cast<llama_word *>(chunk + 4 * kHD + h * 8)[1] = 0;
  }
  reinterpret_cast<llama_word *>(row + 2 * kHD)[0] =
      __builtin_bit_cast(int32_t, scale[4]);
  reinterpret_cast<llama_word *>(row + 2 * kHD)[1] =
      __builtin_bit_cast(int32_t, scale[5]);
  return true;
}

// The decode's rope_q_requant4 and llama_kv_row_quant on one KV group.
__attribute__((noinline)) static void qkv_group(const float *restrict q,
                                                const int8_t *restrict cs,
                                                int8_t *restrict chunk,
                                                int8_t *restrict row) {
  rope_q_requant4(q, reinterpret_cast<const uint16_t *>(cs), chunk);
  alignas(64) int8_t kvfp[2 * kHD * 4 + 2 * kHD * 2];
  copy_words<2 * kHD * 4>(kvfp, q + 4 * kHD);
  copy_words<2 * kHD * 2>(kvfp + 2 * kHD * 4, cs);
  llama_kv_row_quant<kHD>(kvfp, row);
}

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
  RopeFactors rf;
  ExpRange cr = exp_range_empty();
  for (int i = 0; i < 4; i++) {
    rf.f[i] = bf16_factor(
        ::aie::load_v<32>(reinterpret_cast<const int16_t *>(cs) + 32 * i));
    exp_range_add(cr, rf.f[i].h);
  }
  exp_range_get(cr, rf.lo, rf.hi);
  ::aie::accum<acc64, 32> init;
  init.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 8);
  for (int g = 0; g < kGroups; g++) {
    const float *q = fp + g * kGroupFp;
    int8_t *chunk = chunks + g * kChunkStride;
    alignas(64) int8_t row[kRowBytes + 56];
    if (!qkv_group_fast(q, rf, init, chunk, row))
      qkv_group(q, cs, chunk, row);
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
