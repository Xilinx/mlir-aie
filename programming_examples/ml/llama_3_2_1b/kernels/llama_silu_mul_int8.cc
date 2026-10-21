//===- llama_silu_mul_int8.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama silu_mul: SiLU(gate) * up in fp32 from the fp32 gate and up, then
// requantized to int8 by its own absmax for down.
//
//   e = exp_nonpos(-|g|), r = 1 / (1 + e)
//   silu(g) = g * r for g >= 0, g * (e * r) below
//   h = silu(g) * u
//   out = round(h * 127 / max|h|), with the scale max|h| / 127 after it
//
// A row arrives as kCols / kChunk chunks of up, then as many of gate; up is
// kept in h, and each gate chunk turns its part of h into the product.
// llama_oracle.silu_mul mirrors it: products round once (vec_mul_rne), sums
// are vadd.f, and 1 / (1 + e) is a Newton reciprocal from a linear guess.
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

static constexpr int kLanes = 16;
static constexpr int kCols = LLAMA_SILU_MUL_COLS;
static constexpr int kChunk = LLAMA_SILU_CHUNK;
static_assert(kChunk % kLanes == 0 && kCols % kChunk == 0,
              "LLAMA_SILU_CHUNK must be a multiple of 16 dividing the columns");

// 1 / d for d in [1, 2]: the guess 24/17 - 8/17 d, within 1/17, then three
// steps y += y (1 - d y).
static inline ::aie::vector<float, kLanes>
recip_unit(::aie::vector<float, kLanes> d) {
  const ::aie::vector<float, kLanes> one =
      ::aie::broadcast<float, kLanes>(1.0f);
  ::aie::vector<float, kLanes> y =
      ::aie::sub(::aie::broadcast<float, kLanes>(24.0f / 17.0f),
                 vec_mul_rne(::aie::broadcast<float, kLanes>(8.0f / 17.0f), d));
  for (int i = 0; i < 3; i++)
    y = ::aie::add(y, vec_mul_rne(y, ::aie::sub(one, vec_mul_rne(d, y))));
  return y;
}

extern "C" {

// Up's chunk `chunk` of a row into h.
void llama_silu_take_up(float *restrict up, float *restrict h, int32_t chunk) {
  float *dst = h + chunk * kChunk;
  for (int c = 0; c < kChunk / kLanes; c++)
    ::aie::store_v(dst + c * kLanes, ::aie::load_v<kLanes>(up + c * kLanes));
}

// h = SiLU(gate) * h over the row's chunk `chunk`.
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
    ::aie::vector<float, kLanes> e = vec_exp_nonpos(
        ::aie::select(::aie::sub(::aie::zeros<float, kLanes>(), g), g, neg));
    ::aie::vector<float, kLanes> r =
        recip_unit(::aie::add(::aie::broadcast<float, kLanes>(1.0f), e));
    ::aie::vector<float, kLanes> s = ::aie::select(r, vec_mul_rne(e, r), neg);
    ::aie::store_v(
        hc + c * kLanes,
        vec_mul_rne(vec_mul_rne(g, s), ::aie::load_v<kLanes>(hc + c * kLanes)));
  }
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// out = h requantized to int8 by its absmax, out[kCols..kCols + 4) its scale
// (down's act_scale) and out[kCols + 4..kCols + 8) zero.
void llama_silu_quant(float *restrict h, int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  // |v| as int32 bits: for non-NaN floats their integer order is the order
  // of the magnitudes.
  ::aie::vector<int32, kLanes> mag = ::aie::zeros<int32, kLanes>();
  for (int c = 0; c < kCols / kLanes; c++)
    mag = ::aie::max(
        mag,
        ::aie::bit_and(
            int32(0x7fffffff),
            ::aie::load_v<kLanes>(h + c * kLanes).template cast_to<int32>()));
  float absmax = __builtin_bit_cast(float, ::aie::reduce_max(mag));
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float out_scale = absmax * (1.0f / 127.0f);
  ::aie::vector<float, kLanes> inv_v =
      ::aie::broadcast<float, kLanes>(recip_rne(out_scale));
  for (int c = 0; c < kCols / kLanes; c++)
    ::aie::store_v(out + c * kLanes,
                   vec_round_to_i8(vec_mul_rne(
                       ::aie::load_v<kLanes>(h + c * kLanes), inv_v)));
  __builtin_memcpy(out + kCols, &out_scale, 4);
  int32_t zero = 0;
  __builtin_memcpy(out + kCols + 4, &zero, 4);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

} // extern "C"
