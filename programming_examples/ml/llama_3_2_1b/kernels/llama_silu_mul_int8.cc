//===- llama_silu_mul_int8.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama silu_mul: h = SiLU(g) * u in fp32, requantized to int8 by max|h|.
// SiLU(g) = g / (1 + e) for g >= 0, else g e / (1 + e), e = exp_nonpos(-|g|).
// A row arrives as all its gate chunks, then its up chunks, so SiLU overlaps
// the array's up. llama_oracle.silu_mul mirrors every rounding.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_SILU_MUL_COLS
#define LLAMA_SILU_MUL_COLS 8192
#endif
#ifndef LLAMA_SILU_CHUNK
#define LLAMA_SILU_CHUNK 1024
#endif

static constexpr int kLanes = 32;
static constexpr int kCols = LLAMA_SILU_MUL_COLS;
static constexpr int kChunk = LLAMA_SILU_CHUNK;
static_assert(kChunk % kLanes == 0 && kCols % kChunk == 0,
              "LLAMA_SILU_CHUNK must be a multiple of 32 dividing the columns");

// 1 / d rounded to nearest even for d in [1, 2]: three fixed-point Newton
// steps (Q30) from 24/17 - 8/17 d come within one of the rounded significand
// for every such d, which vec_recip_rne_bits then corrects.
static inline ::aie::vector<float, kLanes>
recip_unit(::aie::vector<float, kLanes> d) {
  const ::aie::vector<int32, kLanes> db = d.template cast_to<int32>();
  const ::aie::vector<int32, kLanes> x = ::aie::select(
      ::aie::upshift(
          ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), db)),
          6),
      ::aie::broadcast<int32, kLanes>(1 << 30),
      ::aie::eq(db, int32(0x40000000)));
  ::aie::vector<int32, kLanes> y =
      ::aie::sub(::aie::broadcast<int32, kLanes>(1515870810),
                 ::aie::mul(x, ::aie::broadcast<int32, kLanes>(505290270))
                     .template to_vector<int32>(29));
  for (int i = 0; i < 3; i++) {
    const ::aie::vector<int32, kLanes> t =
        ::aie::mul(x, y).template to_vector<int32>(29);
    // 2^31 - t, which is below 2^31, in wrapping int32 arithmetic.
    y = ::aie::mul(y, ::aie::sub(::aie::broadcast<int32, kLanes>(INT32_MIN), t))
            .template to_vector<int32>(30);
  }
  ::aie::accum<acc64, kLanes> ya;
  ya.from_vector(y, 0);
  return vec_recip_rne_bits(
             db, ::aie::add(ya.template to_vector<int32>(6), int32(125 << 23)))
      .template cast_to<float>();
}

extern "C" {

// h = SiLU(gate) over the row's chunk `chunk`.
void llama_silu_gate(float *restrict gate, float *restrict h, int32_t chunk) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  float *hc = h + chunk * kChunk;
  for (int c = 0; c < kChunk / kLanes; c++) {
    ::aie::vector<float, kLanes> g = ::aie::load_v<kLanes>(gate + c * kLanes);
    // -|g| as 0 - g where g is not negative: OR-ing in the sign bit folds
    // to a vector fneg that Peano cannot legalize on AIE2P.
    auto neg = ::aie::lt(g.template cast_to<int32>(), int32(0));
    ::aie::vector<float, kLanes> e = vec_exp_nonpos<kLanes>(
        ::aie::select(::aie::sub(::aie::zeros<float, kLanes>(), g), g, neg));
    ::aie::vector<float, kLanes> r =
        recip_unit(::aie::add(::aie::broadcast<float, kLanes>(1.0f), e));
    ::aie::vector<float, kLanes> s = ::aie::select(r, vec_mul_rne(e, r), neg);
    ::aie::store_v(hc + c * kLanes, vec_mul_rne(g, s));
  }
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// h = h * up over the row's chunk `chunk`.
void llama_silu_up(float *restrict up, float *restrict h, int32_t chunk) {
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  float *hc = h + chunk * kChunk;
  for (int c = 0; c < kChunk / kLanes; c++)
    ::aie::store_v(hc + c * kLanes,
                   vec_mul_rne(::aie::load_v<kLanes>(hc + c * kLanes),
                               ::aie::load_v<kLanes>(up + c * kLanes)));
  ::aie::set_rounding(prev_rnd);
}

// out = h requantized to int8 by its absmax, out[kCols..kCols + 4) its scale
// (down's act_scale) and out[kCols + 4..kCols + 8) zero.
void llama_silu_quant(float *restrict h, int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  ::aie::vector<int32, kLanes> mag = ::aie::zeros<int32, kLanes>();
  for (int c = 0; c < kCols / kLanes; c++)
    mag = ::aie::max(
        mag,
        ::aie::bit_and(
            int32(0x7fffffff),
            ::aie::load_v<kLanes>(h + c * kLanes).template cast_to<int32>()));
  const float absmax = __builtin_bit_cast(
      float, std::max(::aie::reduce_max(mag), kAbsmaxFloorBits));
  float out_scale = scalar_mul_rne(absmax, 1.0f / 127.0f);
  // |h| <= absmax, so |h * inv| stays near 127, as vec_requant_i8 needs.
  const RequantFactor inv = requant_factor(recip_rne(out_scale));
  for (int c = 0; c < kCols / kLanes; c++)
    ::aie::store_v(
        out + c * kLanes,
        vec_requant_i8(
            ::aie::load_v<kLanes>(reinterpret_cast<const int32_t *>(h) +
                                  c * kLanes),
            inv));
  __builtin_memcpy(out + kCols, &out_scale, 4);
  int32_t zero = 0;
  __builtin_memcpy(out + kCols + 4, &zero, 4);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

} // extern "C"
