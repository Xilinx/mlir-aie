//===- llama_gemv_array.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array's row arithmetic, shared by the decode cores
// (llama_gemv_array.cc) and the M-row prefill cores (llama_gemv_prefill.cc):
// a slot is [kRows*K int8 weights | kRows int32 bias | kRows fp32 w_scales]
// and an activation carries its act_scale in act[K..K+4].
//===----------------------------------------------------------------------===//

#ifndef LLAMA_GEMV_ARRAY_H
#define LLAMA_GEMV_ARRAY_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

template <int kK, int kRows>
static inline int32_t gemv_row_sum(const int8_t *restrict act,
                                   const int8_t *restrict w_row, int32_t bias) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0, "K must be a multiple of 64");
  aie::accum<acc32, kVec> acc;
  acc.from_vector(aie::zeros<int32, kVec>());
  for (int g = 0; g < kK / kVec; g++) {
    aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
    aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
    acc = aie::mac(acc, w_v, x_v);
  }
  return aie::reduce_add(acc.template to_vector<int32>()) + bias;
}

// The f32 epilogue on precomputed row sums (bias included).
template <int kRows>
static inline void epilogue_f32(const int32_t *restrict sums, float act_scale,
                                const float *restrict w_scales,
                                float *restrict out) {
  if (vec_epilogue_exact<80, 174, kRows>(sums, act_scale, w_scales)) {
    ::aie::vector<float, 16> fp =
        vec_scaled_sums<kRows>(sums, act_scale, w_scales);
    for (int n = 0; n < kRows; n++)
      out[n] = fp.get(n);
  } else {
    for (int n = 0; n < kRows; n++)
      out[n] = (float)sums[n] * act_scale * w_scales[n];
  }
}

template <int kK, int kRows>
static inline void gemv_f32(const int8_t *restrict act,
                            const int8_t *restrict w, float *restrict out) {
  float act_scale = *reinterpret_cast<const float *>(act + kK);
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + kRows * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(w + kRows * kK + kRows * 4);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  int32_t sums[kRows];
  for (int n = 0; n < kRows; n++)
    sums[n] = gemv_row_sum<kK, kRows>(act, w + n * kK, bias[n]);
  epilogue_f32<kRows>(sums, act_scale, w_scales, out);
}

// The int8 (gate) epilogue on precomputed row sums. See
// gemm_tile_perchan_acttail_impl (llama_gemm_int8_srs_tiled_layer.cc) for the
// guards around the vector path.
template <int kRows>
static inline void epilogue_i8(const int32_t *restrict sums, float act_scale,
                               const float *restrict w_scales,
                               int8_t *restrict out, float inv_out_scale) {
  if (vec_epilogue_exact<96, 158, kRows>(sums, act_scale, w_scales,
                                         inv_out_scale)) {
    ::aie::vector<float, 16> scaled =
        vec_mul_rne(vec_scaled_sums<kRows>(sums, act_scale, w_scales),
                    ::aie::broadcast<float, 16>(inv_out_scale));
    if (::aie::reduce_max(::aie::bit_and(int32(0x7fffffff),
                                         scaled.template cast_to<int32>())) <
        0x4f000000) {
      ::aie::vector<int8, 16> r = vec_round_to_i8(scaled);
      for (int n = 0; n < kRows; n++)
        out[n] = r.get(n);
      return;
    }
  }
  for (int n = 0; n < kRows; n++) {
    float fp = (float)sums[n] * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;
    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out[n] = (int8_t)r;
  }
}

template <int kK, int kRows>
static inline void gemv_i8(const int8_t *restrict act, const int8_t *restrict w,
                           int8_t *restrict out, float inv_out_scale) {
  float act_scale = *reinterpret_cast<const float *>(act + kK);
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + kRows * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(w + kRows * kK + kRows * 4);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  int32_t sums[kRows];
  for (int n = 0; n < kRows; n++)
    sums[n] = gemv_row_sum<kK, kRows>(act, w + n * kK, bias[n]);
  epilogue_i8<kRows>(sums, act_scale, w_scales, out, inv_out_scale);
}

#endif // LLAMA_GEMV_ARRAY_H
