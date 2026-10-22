//===- llama_vec_fp.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Vector forms of the GEMV kernels' scalar fp32 epilogues, bit-identical to
// the scalar soft-float code where the guards below hold. Needs
// rounding_mode::conv_even and saturation_mode::saturate (set by the caller).
//===----------------------------------------------------------------------===//

#ifndef LLAMA_VEC_FP_H
#define LLAMA_VEC_FP_H

#include <aie_api/aie.hpp>
#include <stdint.h>

// Word loops for copies a __builtin_memcpy would make a call to libc's memcpy
// (Peano passes -fno-builtin-memcpy), which carries no .stack_sizes. Both
// pointers are 4-byte aligned.
using llama_word = int32_t __attribute__((may_alias));

template <int kBytes>
static inline void copy_words(void *restrict dst, const void *restrict src) {
  static_assert(kBytes % 4 == 0);
  for (int i = 0; i < kBytes / 4; ++i)
    ((llama_word *)dst)[i] = ((const llama_word *)src)[i];
}

template <int kBytes>
static inline void zero_words(void *restrict dst) {
  static_assert(kBytes % 4 == 0);
  for (int i = 0; i < kBytes / 4; ++i)
    ((llama_word *)dst)[i] = 0;
}

// Lane-wise a * b rounded to nearest-even, as the scalar soft-float `*` is.
// aie::mul on fp32 vectors is an emulation that is one ulp off on up to ~1.5%
// of lanes, so the 24-bit significands are multiplied exactly into acc64 and
// rounded once by srs. Needs rounding_mode::conv_even (set by the caller).
// Overflow gives +-inf; zero or denormal operands and products below
// FLT_MIN give +-0 where IEEE would keep a denormal.
template <unsigned kN>
static inline ::aie::vector<float, kN> vec_mul_rne(::aie::vector<float, kN> a,
                                                   ::aie::vector<float, kN> b) {
  ::aie::vector<int32, kN> ai = a.template cast_to<int32>();
  ::aie::vector<int32, kN> bi = b.template cast_to<int32>();
  ::aie::vector<int32, kN> ea =
      ::aie::bit_and(int32(0xff), ::aie::downshift(ai, 23));
  ::aie::vector<int32, kN> eb =
      ::aie::bit_and(int32(0xff), ::aie::downshift(bi, 23));
  ::aie::accum<acc64, kN> p = ::aie::mul(
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), ai)),
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bi)));
  // p is in [2^46, 2^48). A rounded p >> 23 reaching 2^24 means p >= 2^47,
  // or p rounds up to 2^47 where both shifts give the same float.
  ::aie::vector<int32, kN> lo = p.template to_vector<int32>(23);
  ::aie::vector<int32, kN> hi = p.template to_vector<int32>(24);
  auto wide = ::aie::ge(lo, int32(1 << 24));
  ::aie::vector<int32, kN> e = ::aie::add(
      ::aie::add(ea, eb), ::aie::select(int32(-127), int32(-126), wide));
  // The significand's implicit bit carries into the exponent field.
  ::aie::vector<int32, kN> mag = ::aie::add(
      ::aie::upshift(::aie::sub(e, int32(1)), 23), ::aie::select(lo, hi, wide));
  mag = ::aie::select(mag, ::aie::broadcast<int32, kN>(0x7f800000),
                      ::aie::ge(e, int32(255)));
  mag = ::aie::select(mag, ::aie::zeros<int32, kN>(),
                      ::aie::lt(::aie::min(::aie::min(ea, eb), e), int32(1)));
  return ::aie::bit_or(
             ::aie::bit_and(int32(0x80000000), ::aie::bit_xor(ai, bi)), mag)
      .template cast_to<float>();
}

// a * b rounded to nearest even, as the scalar soft-float `*` is -- the
// scalar form of vec_mul_rne, for callers with one pair of floats rather than
// a vector. The target has no scalar float multiplier (`*` on two floats is
// __mulsf3), so this avoids that libcall via the same bit-exact machinery.
// 8 lanes: the narrowest width AIE2P legalizes an integer bit_and/bit_or at.
__attribute__((noinline)) static float scalar_mul_rne(float a, float b) {
  return vec_mul_rne(::aie::broadcast<float, 8>(a), ::aie::broadcast<float, 8>(b))
      .get(0);
}

// Lane-wise round_to_i8 -- (int32)(v + (v >= 0 ? 0.5 : -0.5)) clamped to
// int8 -- bit-identical to the scalar form, whose `+` is the same vadd.f.
// float2fix gives floor(t) or ceil(t), not always the nearest; it rounded
// away from zero exactly when |n| > |t|. Needs saturation_mode::saturate for
// the clamp. The +-0.5 is an integer select because OR-ing in v's sign bit
// folds to a vector copysign that Peano cannot legalize on AIE2P.
template <unsigned kN>
static inline ::aie::vector<int8, kN>
vec_round_to_i8(::aie::vector<float, kN> v) {
  ::aie::vector<int32, kN> half =
      ::aie::select(::aie::broadcast<int32, kN>(0x3f000000),
                    ::aie::broadcast<int32, kN>(int32(0xbf000000)),
                    ::aie::lt(v.template cast_to<int32>(), int32(0)));
  ::aie::vector<float, kN> t = ::aie::add(v, half.template cast_to<float>());
  ::aie::vector<int32, kN> n = ::aie::to_fixed<int32>(t, 0);
  ::aie::vector<int32, kN> n_bits =
      ::aie::to_float<float>(n, 0).template cast_to<int32>();
  auto away =
      ::aie::lt(::aie::bit_and(int32(0x7fffffff), t.template cast_to<int32>()),
                ::aie::bit_and(int32(0x7fffffff), n_bits));
  ::aie::vector<int32, kN> toward_zero = ::aie::select(
      ::aie::add(n, int32(1)), ::aie::sub(n, int32(1)), ::aie::gt(n, int32(0)));
  ::aie::accum<acc32, kN> acc;
  acc.from_vector(::aie::select(n, toward_zero, away));
  return acc.template to_vector<int8>(0);
}

// 1 / a rounded to nearest even, as IEEE division rounds it (checked against
// it for every positive normal a below 2^126, the domain it needs), by
// restoring division of the mantissa: __divsf3 overflows the decode cores'
// program memory.
static inline float recip_rne(float a) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, a);
  const uint32_t e = bits >> 23;
  const uint32_t m = (bits & 0x7fffffu) | 0x800000u;
  uint32_t rem = 1u << 24, q = 0;
  for (int i = 0; i < 24; i++) {
    q <<= 1;
    if (rem >= m) {
      rem -= m;
      q |= 1;
    }
    rem <<= 1;
  }
  if (q == 1u << 23 && rem == 0)
    return __builtin_bit_cast(float, (254u - e) << 23);
  if (rem > m || (rem == m && (q & 1)))
    q++;
  return __builtin_bit_cast(float, ((253u - e) << 23) + (q - (1u << 23)));
}

template <unsigned kN>
static inline ::aie::vector<float, kN> vec_recip(::aie::vector<float, kN> a) {
  alignas(64) float lanes[kN];
  ::aie::store_v(lanes, a);
  for (unsigned i = 0; i < kN; i++)
    lanes[i] = recip_rne(lanes[i]);
  return ::aie::load_v<kN>(lanes);
}

// e^x for x <= 0 lane-wise, as llama_oracle.exp_nonpos: with
// t = x * log2(e) (vec_mul_rne), k = trunc(t) and f = t - k in (-1, 0],
// e = 2^k * 2^f where 2^f = 2^-g, g = -f rounded to Q23 by the vadd.f that
// forms 1 - f, is a degree-6 Q29 polynomial in Horner form whose products
// round half to even (acc64 srs). 0 below t = -125. float2fix gives floor(t)
// or ceil(t); trunc(t) for t <= 0 is the ceiling, so a negative n below t
// (compared as sign-magnitude bits) is bumped by one.
template <unsigned kN = 16>
__attribute__((noinline)) static ::aie::vector<float, kN>
vec_exp_nonpos(::aie::vector<float, kN> x) {
  static constexpr int32_t kC[] = {-684167,   5142538,    -29791239,
                                   128969442, -372130494, 536870912};
  ::aie::vector<float, kN> t =
      vec_mul_rne(x, ::aie::broadcast<float, kN>(1.442695022f));
  ::aie::vector<int32, kN> tb = t.template cast_to<int32>();
  ::aie::vector<int32, kN> n = ::aie::to_fixed<int32>(t, 0);
  ::aie::vector<int32, kN> nb =
      ::aie::to_float<float>(n, 0).template cast_to<int32>();
  ::aie::vector<int32, kN> k = ::aie::select(
      n, ::aie::add(n, int32(1)), ::aie::gt(nb, tb) & ::aie::lt(nb, int32(0)));
  ::aie::vector<float, kN> f = ::aie::sub(t, ::aie::to_float<float>(k, 0));
  ::aie::vector<int32, kN> g =
      ::aie::sub(::aie::sub(::aie::broadcast<float, kN>(1.0f), f)
                     .template cast_to<int32>(),
                 int32(0x3f800000));
  ::aie::vector<int32, kN> p = ::aie::broadcast<int32, kN>(58465);
  for (int i = 0; i < 6; i++)
    p = ::aie::add(::aie::mul(p, g).template to_vector<int32>(23), kC[i]);
  return ::aie::select(
             ::aie::add(::aie::to_float<float>(p, 29).template cast_to<int32>(),
                        ::aie::upshift(k, 23)),
             ::aie::zeros<int32, kN>(),
             ::aie::lt(tb, int32(0)) &
                 ::aie::gt(tb, __builtin_bit_cast(int32_t, -125.0f)))
      .template cast_to<float>();
}

// Per-head requant of kHeads heads of kHD fp32 values:
// each head's scale is max(absmax, 1e-12) * (1/127), its body int8
// round(q * (1 / scale)); tail[h*8..+4] gets the scale, [+4..+8] zero.
template <int kHeads, int kHD>
static inline void q_requant_heads(const float *restrict q_fp,
                                   int8_t *restrict body,
                                   int8_t *restrict tail) {
  constexpr int kV = 16;
  // One group covers every head in one vec_recip call; vec_recip's per-lane
  // recip_rne is scalar, so padding the group to kV would spend most of its
  // cost on unused lanes. kGroup stays at 8 (not kHeads) below 8 heads: AIE2P
  // doesn't legalize a 128-bit (4-lane) integer bit_and/bit_or.
  constexpr int kGroup = kHeads < 8 ? 8 : kHeads;
  static_assert(kGroup <= kV);
  alignas(64) float q_scale[kGroup];
  alignas(64) float q_inv[kGroup];
  for (int h = kHeads; h < kGroup; h++)
    q_scale[h] = 0.0f;
  for (int h = 0; h < kHeads; h++) {
    const float *src = q_fp + h * kHD;
    ::aie::vector<int32, kV> mag = ::aie::zeros<int32, kV>();
    for (int c = 0; c < kHD / kV; c++)
      mag = ::aie::max(
          mag, ::aie::bit_and(
                   int32(0x7fffffff),
                   ::aie::load_v<kV>(src + c * kV).template cast_to<int32>()));
    q_scale[h] = __builtin_bit_cast(float, ::aie::reduce_max(mag));
  }
  // Every head's scale and reciprocal at once. max(absmax, 1e-12f) is an
  // integer max of the non-negative bits.
  ::aie::vector<float, kGroup> absmax =
      ::aie::max(
          ::aie::load_v<kGroup>(q_scale).template cast_to<int32>(),
          int32(0x2b8cbccc))
          .template cast_to<float>();
  ::aie::vector<float, kGroup> scale =
      vec_mul_rne(absmax, ::aie::broadcast<float, kGroup>(1.0f / 127.0f));
  ::aie::store_v(q_scale, scale);
  ::aie::store_v(q_inv, vec_recip(scale));
  for (int h = 0; h < kHeads; h++) {
    const float *src = q_fp + h * kHD;
    int8_t *dst = body + h * kHD;
    ::aie::vector<float, kV> q_inv_v = ::aie::broadcast<float, kV>(q_inv[h]);
    for (int c = 0; c < kHD / kV; c++)
      ::aie::store_v(dst + c * kV,
                     vec_round_to_i8(vec_mul_rne(
                         ::aie::load_v<kV>(src + c * kV), q_inv_v)));
    __builtin_memcpy(tail + h * 8 + 0, &q_scale[h], 4);
    int32_t zero = 0;
    __builtin_memcpy(tail + h * 8 + 4, &zero, 4);
  }
}

// True when the vector epilogue equals the scalar soft-float one: every
// |sum| < 2^23, since fx2flt misrounds odd sums below -(2^24 - 2^16), and the
// exponent fields of act_scale, every w_scale and out_scale in [kLo, kHi],
// chosen by the caller so that each product of its chain stays normal.
template <int kLo, int kHi, int kNTile>
static inline bool vec_epilogue_exact(const int32_t *sums, float act_scale,
                                      const float *w_scales,
                                      float out_scale = 1.0f) {
  bool ok = true;
  for (int n = 0; n < kNTile + 2; n++) {
    float f = n < kNTile ? w_scales[n] : n == kNTile ? act_scale : out_scale;
    ok &= (uint32_t)(((__builtin_bit_cast(int32_t, f) >> 23) & 0xff) - kLo) <=
          kHi - kLo;
  }
  for (int n = 0; n < kNTile; n++)
    ok &= (uint32_t)sums[n] + (1u << 23) - 1 < (1u << 24) - 1;
  return ok;
}

// Lanes n < kNTile hold (float)sums[n] * a * w[n], in the scalar code's
// order; the other lanes hold +0.
template <int kNTile>
static inline ::aie::vector<float, 16>
vec_scaled_sums(const int32_t *sums, float a, const float *w) {
  ::aie::vector<int32, 16> sum_v = ::aie::zeros<int32, 16>();
  ::aie::vector<float, 16> w_v = ::aie::zeros<float, 16>();
  for (int n = 0; n < kNTile; n++) {
    sum_v.set(sums[n], n);
    w_v.set(w[n], n);
  }
  return vec_mul_rne(vec_mul_rne(::aie::to_float<float>(sum_v, 0),
                                 ::aie::broadcast<float, 16>(a)),
                     w_v);
}

#endif // LLAMA_VEC_FP_H
