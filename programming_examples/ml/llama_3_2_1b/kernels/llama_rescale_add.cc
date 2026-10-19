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

#ifndef LLAMA_RESCALE_D
#define LLAMA_RESCALE_D 2048
#endif

// One 512-bit fp32 register; wider vectors made Peano spill to 3 KB frames.
static constexpr int kVec = 16;

// Lane-wise round_to_i8 -- (int32)(v + (v >= 0 ? 0.5 : -0.5)) clamped to
// int8 -- bit-identical to the scalar form, whose `+` is the same vadd.f.
// float2fix gives floor(t) or ceil(t), not always the nearest; it rounded
// away from zero exactly when |n| > |t|. Needs saturation_mode::saturate for
// the clamp. The +-0.5 is an integer select because OR-ing in v's sign bit
// folds to a vector copysign that Peano cannot legalize on AIE2P.
static inline ::aie::vector<int8, kVec>
vec_round_to_i8(::aie::vector<float, kVec> v) {
  ::aie::vector<int32, kVec> half =
      ::aie::select(::aie::broadcast<int32, kVec>(0x3f000000),
                    ::aie::broadcast<int32, kVec>(int32(0xbf000000)),
                    ::aie::lt(v.template cast_to<int32>(), int32(0)));
  ::aie::vector<float, kVec> t = ::aie::add(v, half.template cast_to<float>());
  ::aie::vector<int32, kVec> n = ::aie::to_fixed<int32>(t, 0);
  ::aie::vector<int32, kVec> n_bits =
      ::aie::to_float<float>(n, 0).template cast_to<int32>();
  auto away =
      ::aie::lt(::aie::bit_and(int32(0x7fffffff), t.template cast_to<int32>()),
                ::aie::bit_and(int32(0x7fffffff), n_bits));
  ::aie::vector<int32, kVec> toward_zero = ::aie::select(
      ::aie::add(n, int32(1)), ::aie::sub(n, int32(1)), ::aie::gt(n, int32(0)));
  ::aie::accum<acc32, kVec> acc;
  acc.from_vector(::aie::select(n, toward_zero, away));
  return acc.template to_vector<int8>(0);
}

// Lane-wise a * b rounded to nearest-even, as the scalar soft-float `*` is.
// aie::mul on fp32 vectors is an emulation that is one ulp off on up to ~1.5%
// of lanes, so the 24-bit significands are multiplied exactly into acc64 and
// rounded once by srs. Needs rounding_mode::conv_even (set by the caller).
// Overflow gives +-inf; zero or denormal operands and products below
// FLT_MIN give +-0 (callers round these to int8 0).
static inline ::aie::vector<float, kVec>
vec_mul_rne(::aie::vector<float, kVec> a, ::aie::vector<float, kVec> b) {
  ::aie::vector<int32, kVec> ai = a.template cast_to<int32>();
  ::aie::vector<int32, kVec> bi = b.template cast_to<int32>();
  ::aie::vector<int32, kVec> ea =
      ::aie::bit_and(int32(0xff), ::aie::downshift(ai, 23));
  ::aie::vector<int32, kVec> eb =
      ::aie::bit_and(int32(0xff), ::aie::downshift(bi, 23));
  ::aie::accum<acc64, kVec> p = ::aie::mul(
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), ai)),
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bi)));
  // p is in [2^46, 2^48). A rounded p >> 23 reaching 2^24 means p >= 2^47,
  // or p rounds up to 2^47 where both shifts give the same float.
  ::aie::vector<int32, kVec> lo = p.template to_vector<int32>(23);
  ::aie::vector<int32, kVec> hi = p.template to_vector<int32>(24);
  auto wide = ::aie::ge(lo, int32(1 << 24));
  ::aie::vector<int32, kVec> e = ::aie::add(
      ::aie::add(ea, eb), ::aie::select(int32(-127), int32(-126), wide));
  // The significand's implicit bit carries into the exponent field.
  ::aie::vector<int32, kVec> mag = ::aie::add(
      ::aie::upshift(::aie::sub(e, int32(1)), 23), ::aie::select(lo, hi, wide));
  mag = ::aie::select(mag, ::aie::broadcast<int32, kVec>(0x7f800000),
                      ::aie::ge(e, int32(255)));
  mag = ::aie::select(mag, ::aie::zeros<int32, kVec>(),
                      ::aie::lt(::aie::min(::aie::min(ea, eb), e), int32(1)));
  return ::aie::bit_or(
             ::aie::bit_and(int32(0x80000000), ::aie::bit_xor(ai, bi)), mag)
      .template cast_to<float>();
}

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
