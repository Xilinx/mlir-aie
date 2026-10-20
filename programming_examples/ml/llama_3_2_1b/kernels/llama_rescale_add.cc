//===- llama_rescale_add.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Residual rescale-add: adds an int8 residual (carrying a per-token fp32
// scale) to the TRUE fp32 projection output, requanting to a NEW
// per-token dynamic scale recomputed at every add -- replaces a
// catastrophic static-0.05 residual scale (numpy_layer_mh.py's
// residual_dyn add). Two-pass, no intermediate buffer (Peano AIE2P
// corrupts stack fp32 arrays kept across loops): pass A finds absmax,
// pass B requants, recomputing the add both times (negligible cost vs.
// the gemms feeding this). Buffers: resid/out int8[kD+8] (4 B scale + 4
// B pad); proj float[kD].
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_RESCALE_D
#define LLAMA_RESCALE_D 2048
#endif

// One 512-bit fp32 register; wider vectors made Peano spill to 3 KB frames.
static constexpr int kVec = 16;

// |v| as int32 bits: for non-NaN floats their integer order is the order of
// the magnitudes, so absmax is an integer max (no emulated f32 compare) and
// one horizontal reduction at the end.
static inline ::aie::vector<int32, kVec>
vec_magnitude_bits(::aie::vector<float, kVec> v) {
  return ::aie::bit_and(int32(0x7fffffff), v.template cast_to<int32>());
}

static inline float bits_to_float(int32_t bits) {
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// x_fp = resid_i8 * resid_scale + proj, kVec lanes at a time: the multiply
// and the add round separately, as the scalar form does, whose `+` is vadd.f.
static inline ::aie::vector<float, kVec>
vec_residual_sum(const int8_t *restrict resid, const float *restrict proj,
                 ::aie::vector<float, kVec> resid_scale_v) {
  return ::aie::add(
      vec_mul_rne(::aie::to_float<float>(::aie::load_v<kVec>(resid)),
                  resid_scale_v),
      ::aie::load_v<kVec>(proj));
}

// Software fp32 reciprocal (see llama_flowkv.cc for why Peano's `/` needs
// this workaround); identical to the other kernels' sw_recip so the numpy
// golden and device target share the function.
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

extern "C" {

void llama_rescale_add_D(int8_t *restrict resid, float *restrict proj,
                         int8_t *restrict out) {
  event0();

  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  constexpr int kD = LLAMA_RESCALE_D;
  static_assert(kD % kVec == 0, "LLAMA_RESCALE_D must be a multiple of 16");
  constexpr int kChunks = kD / kVec;

  float resid_scale;
  memcpy(&resid_scale, resid + kD, 4);
  ::aie::vector<float, kVec> resid_scale_v =
      ::aie::broadcast<float, kVec>(resid_scale);

  // Pass A: x_fp = resid_i8 * resid_scale + proj; track absmax.
  ::aie::vector<int32, kVec> mag = ::aie::zeros<int32, kVec>();
  for (int c = 0; c < kChunks; c++)
    mag =
        ::aie::max(mag, vec_magnitude_bits(vec_residual_sum(
                            resid + c * kVec, proj + c * kVec, resid_scale_v)));
  float absmax = bits_to_float(::aie::reduce_max(mag));

  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float x_scale = absmax * (1.0f / 127.0f);
  float inv = sw_recip(x_scale);
  ::aie::vector<float, kVec> inv_v = ::aie::broadcast<float, kVec>(inv);

  // Pass B: recompute the fp32 sum, requant to int8 with the new scale.
  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> xf =
        vec_residual_sum(resid + c * kVec, proj + c * kVec, resid_scale_v);
    ::aie::store_v(out + c * kVec, vec_round_to_i8(vec_mul_rne(xf, inv_v)));
  }

  memcpy(out + kD, &x_scale, 4);
  int32_t zero = 0;
  memcpy(out + kD + 4, &zero, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

} // extern "C"
