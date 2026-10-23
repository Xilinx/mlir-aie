//===- llama_kv_row.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// One token's K/V cache row from its fp32 k and v: k roped, then K and V
// each quantized to int8 with its own absmax / 127 scale (llama_oracle's
// append).
// Needs rounding_mode::conv_even and saturation_mode::saturate (set by the
// caller).
//===----------------------------------------------------------------------===//

#ifndef LLAMA_KV_ROW_H
#define LLAMA_KV_ROW_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

// kvfp_packed is [k fp32[kHD] | v fp32[kHD] | cos bf16[kHD] | sin bf16[kHD]]
// at a 32-byte boundary, so each 16-lane value is two 8-lane loads. row, at a
// 64-byte boundary, gets [k int8[kHD] | v int8[kHD] | k_scale | v_scale].
template <int kHD>
static inline void llama_kv_row_quant(const int8_t *restrict kvfp_packed,
                                      int8_t *restrict row) {
  constexpr int kV = 16;
  constexpr int kHalf = kHD / 2;
  static_assert(kHalf % kV == 0);
  const float *k_fp = reinterpret_cast<const float *>(kvfp_packed);
  const float *v_fp = k_fp + kHD;
  const int16_t *cs_bits =
      reinterpret_cast<const int16_t *>(kvfp_packed + kHD * 8);

  // bf16 to fp32 is the bf16 bits over a zero low half.
  alignas(64) float cs[2 * kHD];
  for (int c = 0; c < kHD / kV; c++) {
    auto [lo, hi] = ::aie::interleave_zip(
        ::aie::zeros<int16, 2 * kV>(),
        ::aie::concat(::aie::load_v<kV>(cs_bits + c * 2 * kV),
                      ::aie::load_v<kV>(cs_bits + c * 2 * kV + kV)),
        1);
    ::aie::store_v(cs + c * 2 * kV, lo.template cast_to<float>());
    ::aie::store_v(cs + c * 2 * kV + kV, hi.template cast_to<float>());
  }

  alignas(64) float k_rope[kHD];
  for (int j = 0; j < kHalf; j += kV) {
    ::aie::vector<float, kV> x1 = ::aie::concat(::aie::load_v<8>(k_fp + j),
                                                ::aie::load_v<8>(k_fp + j + 8));
    ::aie::vector<float, kV> x2 =
        ::aie::concat(::aie::load_v<8>(k_fp + kHalf + j),
                      ::aie::load_v<8>(k_fp + kHalf + j + 8));
    ::aie::store_v(
        k_rope + j,
        ::aie::sub(vec_mul_rne(x1, ::aie::load_v<kV>(cs + j)),
                   vec_mul_rne(x2, ::aie::load_v<kV>(cs + kHD + j))));
    ::aie::store_v(
        k_rope + kHalf + j,
        ::aie::add(vec_mul_rne(x2, ::aie::load_v<kV>(cs + kHalf + j)),
                   vec_mul_rne(x1, ::aie::load_v<kV>(cs + kHD + kHalf + j))));
  }

  ::aie::vector<int32, kV> k_mag = ::aie::zeros<int32, kV>();
  ::aie::vector<int32, kV> v_mag = ::aie::zeros<int32, kV>();
  for (int c = 0; c < kHD / kV; c++) {
    k_mag = ::aie::max(
        k_mag,
        ::aie::bit_and(
            int32(0x7fffffff),
            ::aie::load_v<kV>(k_rope + c * kV).template cast_to<int32>()));
    v_mag = ::aie::max(
        v_mag, ::aie::bit_and(int32(0x7fffffff),
                              ::aie::concat(::aie::load_v<8>(v_fp + c * kV),
                                            ::aie::load_v<8>(v_fp + c * kV + 8))
                                  .template cast_to<int32>()));
  }
  ::aie::vector<int32, 8> absmax = ::aie::zeros<int32, 8>();
  absmax.set(::aie::reduce_max(k_mag), 0);
  absmax.set(::aie::reduce_max(v_mag), 1);
  ::aie::vector<float, 8> scale = vec_mul_rne(
      ::aie::max(absmax, kAbsmaxFloorBits).template cast_to<float>(),
      ::aie::broadcast<float, 8>(1.0f / 127.0f));
  float ks = scale.get(0);
  float vs = scale.get(1);
  ::aie::vector<float, kV> k_inv = ::aie::broadcast<float, kV>(recip_rne(ks));
  ::aie::vector<float, kV> v_inv = ::aie::broadcast<float, kV>(recip_rne(vs));

  for (int c = 0; c < kHD / kV; c++) {
    ::aie::store_v(row + c * kV,
                   vec_round_to_i8(
                       vec_mul_rne(::aie::load_v<kV>(k_rope + c * kV), k_inv)));
    ::aie::store_v(row + kHD + c * kV,
                   vec_round_to_i8(vec_mul_rne(
                       ::aie::concat(::aie::load_v<8>(v_fp + c * kV),
                                     ::aie::load_v<8>(v_fp + c * kV + 8)),
                       v_inv)));
  }
  reinterpret_cast<llama_word *>(row + 2 * kHD)[0] =
      __builtin_bit_cast(int32_t, ks);
  reinterpret_cast<llama_word *>(row + 2 * kHD)[1] =
      __builtin_bit_cast(int32_t, vs);
}

#endif // LLAMA_KV_ROW_H
