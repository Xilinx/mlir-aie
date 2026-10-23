//===- llama_gemv_array.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// What the GEMV array's kernels (llama_gemv_array.cc, llama_gemv_mma.cc) and
// the prefill's (llama_gemm_mma.cc, llama_prefill_glue.cc) share:
// llama_vec_fp.h's fp32 helpers, the mma epilogue and the Q rope.
//===----------------------------------------------------------------------===//

#ifndef LLAMA_GEMV_ARRAY_H
#define LLAMA_GEMV_ARRAY_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

// A KV group's 4 Q heads (q: 4 x 64 fp32) roped in fp32 with a position's
// cos/sin (cs_bits: bf16 cos[64], then sin[64]) as llama_kv_row.h ropes k,
// then requantized per head into out: 256 int8, then per head [scale f32 |
// 0]. Needs rounding_mode::conv_even and saturation_mode::saturate.
static inline void rope_q_requant4(const float *restrict q,
                                   const uint16_t *restrict cs_bits,
                                   int8_t *restrict out) {
  constexpr int kHD = 64;
  constexpr int kHalf = kHD / 2;
  alignas(64) float cs_f[2 * kHD];
  for (int i = 0; i < 2 * kHD; i++)
    cs_f[i] = __builtin_bit_cast(float, (uint32_t)cs_bits[i] << 16);
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
}

// The exponent fields of the scales the epilogue's vector form takes.
static constexpr uint32_t kEpilogueExpLo = 80, kEpilogueExpHi = 174;

// The f32 epilogue's vector form on activations 4h..4h+3's 32 sums (lane
// (m - 4h)*8 + n, act_scales[m - 4h]): adds the 8 rows' biases to sums and,
// when every scale's exponent field is in [kEpilogueExpLo, kEpilogueExpHi]
// and every sum is within +-2^23 (ok), returns the 32 values. A weight slot's
// trailer holds the 8 rows' int32 biases, then their fp32 scales.
static inline ::aie::vector<float, 32>
half_epilogue_vec(::aie::vector<int32, 32> &sums,
                  const float *restrict act_scales,
                  const int8_t *restrict trailer, bool &ok) {
  const int32_t *bias = reinterpret_cast<const int32_t *>(trailer);
  const float *w_scales = reinterpret_cast<const float *>(trailer + 32);
  ok = true;
  for (int i = 0; i < 4 + 8; i++) {
    float f = i < 4 ? act_scales[i] : w_scales[i - 4];
    ok &= (uint32_t)(((__builtin_bit_cast(int32_t, f) >> 23) & 0xff) -
                     kEpilogueExpLo) <= kEpilogueExpHi - kEpilogueExpLo;
  }
  ::aie::vector<float, 8> w8 = ::aie::load_v<8>(w_scales);
  ::aie::vector<float, 32> w_v = ::aie::concat(w8, w8, w8, w8);
  ::aie::vector<float, 32> a_v =
      ::aie::concat(::aie::broadcast<float, 8>(act_scales[0]),
                    ::aie::broadcast<float, 8>(act_scales[1]),
                    ::aie::broadcast<float, 8>(act_scales[2]),
                    ::aie::broadcast<float, 8>(act_scales[3]));
  ::aie::vector<int32, 8> b8 = ::aie::load_v<8>(bias);
  sums = ::aie::add(sums, ::aie::concat(b8, b8, b8, b8));
  ok &=
      (::aie::ge(sums, int32(-(1 << 23) + 1)) & ::aie::lt(sums, int32(1 << 23)))
          .count() == 32;
  ::aie::vector<float, 32> fp;
  if (ok)
    fp = vec_mul_rne(vec_mul_rne(::aie::to_float<float>(sums, 0), a_v), w_v);
  return fp;
}

// The f32 epilogue on activations 4h..4h+3's 32 sums, bias added, to
// out[m*ostride + row0 + n] for the first m_count activations: on 32 lanes
// when half_epilogue_vec's guards hold, else per element.
__attribute__((noinline)) static void
half_epilogue(::aie::vector<int32, 32> sums, const float *restrict act_scales,
              const int8_t *restrict trailer, int8_t *restrict out, int h,
              int32_t ostride, int32_t row0, int32_t m_count) {
  const float *w_scales = reinterpret_cast<const float *>(trailer + 32);
  const int m_end = m_count - 4 * h < 4 ? m_count - 4 * h : 4;
  bool ok;
  ::aie::vector<float, 32> fp =
      half_epilogue_vec(sums, act_scales, trailer, ok);
  for (int m = 0; m < m_end; m++)
    for (int n = 0; n < 8; n++) {
      const int lane = m * 8 + n;
      float out_val;
      if (ok) {
        out_val = fp.get(lane);
      } else {
        out_val = scalar_mul_rne(
            scalar_mul_rne((float)sums.get(lane), act_scales[m]), w_scales[n]);
      }
      reinterpret_cast<float *>(out)[(4 * h + m) * ostride + row0 + n] =
          out_val;
    }
}

#endif // LLAMA_GEMV_ARRAY_H
