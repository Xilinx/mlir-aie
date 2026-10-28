//===- llama_vec_fp.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// fp32 helpers bit-identical to llama_oracle's scalar arithmetic. Needs
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

// 32 significands below 2^24 as the 16-bit planes (low halves, high halves)
// that the target's 32-bit multiplies split their operands into.
struct Sig24 {
  v32uint16 lo;
  v32int16 hi;
};

static inline Sig24 sig24_planes(::aie::vector<int32, 32> m) {
  return {(v32uint16)::shuffle(m.extract<16>(0), m.extract<16>(1), 2),
          (v32int16)::shuffle(m.extract<16>(0), m.extract<16>(1), 3)};
}

// The exact product of two significands given as planes (plus acc with kAdd):
// the four 16-bit partial products as the 32-bit multiply sums them.
template <bool kAdd>
static inline v32acc64 sig24_mac(Sig24 a, Sig24 b, v32acc64 acc) {
  v32acc64 p = ::mul_elem_32_conf(a.hi, 0, b.hi, 0, 0);
  p = ::mac_elem_32_conf(a.hi, 0, b.lo, 0, p, 0, 1, 0, 0);
  p = ::mac_elem_32_conf(a.lo, 0, b.hi, 0, p, 0, 0, 0, 0);
  if constexpr (kAdd)
    return ::addmac_elem_32_conf(a.lo, 0, b.lo, 0, p, acc, 0, 1, 0, 0, 0);
  else
    return ::mac_elem_32_conf(a.lo, 0, b.lo, 0, p, 0, 1, 0, 0);
}

// A significand's planes times bf16 significands (b: 8 bits, as int16
// lanes), exactly, plus acc: two partial products.
static inline v32acc64 sig24x8_mac(Sig24 a, v32int16 b, v32acc64 acc) {
  v32acc64 p = ::mul_elem_32_conf(a.hi, 0, b, 0, 0);
  return ::addmac_elem_32_conf(a.lo, 0, b, 0, p, acc, 0, 1, 0, 0, 0);
}

// The nonzero exponent fields of fp32 or bf16 lanes seen so far, as E << 7
// (their high 16 bits masked to the exponent): lo_m1 is min - 1, so a zero's
// wraps above every exponent.
struct ExpRange {
  ::aie::vector<uint16, 32> lo_m1;
  ::aie::vector<int16, 32> hi;
};

static inline ExpRange exp_range_empty() {
  return {::aie::broadcast<uint16, 32>(0xffff), ::aie::zeros<int16, 32>()};
}

static inline void exp_range_add(ExpRange &r, ::aie::vector<int16, 32> h) {
  r.lo_m1 = ::aie::min(r.lo_m1, ::aie::sub(h.cast_to<uint16>(), uint16(1)));
  r.hi = ::aie::max(r.hi, h);
}

// The least and greatest exponent field of an ExpRange; lo > hi if empty.
static inline void exp_range_get(const ExpRange &r, int &lo, int &hi) {
  lo = (int)(::aie::reduce_min(r.lo_m1) + 1u) >> 7;
  hi = ::aie::reduce_max(r.hi) >> 7;
}

// 32 fp32 bit patterns as significand planes (implicit bit set) and their
// E << 7 (the high plane's bitwise ops on its int32 view: 16-bit lanes don't
// vectorize them).
struct XPlanes {
  Sig24 sig;
  ::aie::vector<int16, 32> h;
};

static inline XPlanes x_planes(::aie::vector<int32, 32> bits) {
  Sig24 s = sig24_planes(bits);
  ::aie::vector<int32, 16> hv = ::aie::vector<int16, 32>(s.hi).cast_to<int32>();
  s.hi = ::aie::bit_or(int32(0x00800080), ::aie::bit_and(int32(0x007f007f), hv))
             .cast_to<int16>();
  return {s, ::aie::bit_and(int32(0x7f807f80), hv).cast_to<int16>()};
}

// 32 bf16 lanes as a factor of x_bf16_mul: their significands (8 bits) as
// int16 lanes, their sign and exponent bits in fp32 position less 126 << 23,
// and their E << 7.
struct Bf16Factor {
  v32int16 sig;
  ::aie::vector<int32, 32> e;
  ::aie::vector<int16, 32> h;
};

static inline Bf16Factor bf16_factor(::aie::vector<int16, 32> g) {
  const ::aie::vector<int32, 16> gw = g.cast_to<int32>();
  auto [g_lo, g_hi] = ::aie::interleave_zip(::aie::zeros<int16, 32>(), g, 1);
  return {
      ::aie::bit_or(int32(0x00800080), ::aie::bit_and(int32(0x007f007f), gw))
          .cast_to<int16>(),
      ::aie::sub(::aie::bit_and(int32(0xff800000),
                                ::aie::concat(g_lo.cast_to<int32>(),
                                              g_hi.cast_to<int32>())),
                 int32(126 << 23)),
      ::aie::bit_and(int32(0x7f807f80), gw).cast_to<int16>()};
}

// vec_mul_rne(x, g) for x (planes of `xb`) and bf16 g where every product of
// a nonzero x and g is normal (exponent fields in [2, 253] before rounding):
// one exact product of significands rounded once by srs (by 7, or 8 where it
// reaches 2^31), exponents and signs added as integers; lanes where x or g
// has exponent field 0 (zero or denormal, which vec_mul_rne flushes) get +0.
// init is -2^31 in every lane. Needs rounding_mode::conv_even.
static inline ::aie::vector<int32, 32>
x_bf16_mul(::aie::vector<int32, 32> xb, const XPlanes &x, const Bf16Factor &g,
           const ::aie::accum<acc64, 32> &init) {
  ::aie::accum<acc64, 32> p(sig24x8_mac(x.sig, g.sig, init));
  ::aie::vector<int32, 32> lo = p.template to_vector<int32>(7);
  ::aie::vector<int32, 32> hi = p.template to_vector<int32>(8);
  ::aie::vector<int32, 32> v =
      ::aie::add(::aie::add(::aie::bit_and(int32(0xff800000), xb), g.e),
                 ::aie::select(lo, hi, ::aie::ge(lo, int32(0))));
  return ::aie::select(v, ::aie::zeros<int32, 32>(),
                       ::aie::eq(x.h, int16(0)) | ::aie::eq(g.h, int16(0)));
}

// vec_mul_rne with b given as its bits bi, exponent field eb and significand
// with the implicit bit mb (vec_mul_rne's own split of b), for a b that
// several products share.
template <unsigned kN>
static inline ::aie::vector<float, kN>
vec_mul_rne_split(::aie::vector<float, kN> a, ::aie::vector<int32, kN> bi,
                  ::aie::vector<int32, kN> eb, ::aie::vector<int32, kN> mb) {
  ::aie::vector<int32, kN> ai = a.template cast_to<int32>();
  ::aie::vector<int32, kN> ea =
      ::aie::bit_and(int32(0xff), ::aie::downshift(ai, 23));
  ::aie::accum<acc64, kN> p = ::aie::mul(
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), ai)), mb);
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

// Lane-wise a * b rounded to nearest-even, as the scalar soft-float `*` is.
// aie::mul on fp32 vectors is an emulation that is one ulp off on up to ~1.5%
// of lanes, so the 24-bit significands are multiplied exactly into acc64 and
// rounded once by srs. Needs rounding_mode::conv_even (set by the caller).
// Overflow gives +-inf; zero or denormal operands and products below
// FLT_MIN give +-0 where IEEE would keep a denormal.
template <unsigned kN>
static inline ::aie::vector<float, kN> vec_mul_rne(::aie::vector<float, kN> a,
                                                   ::aie::vector<float, kN> b) {
  ::aie::vector<int32, kN> bi = b.template cast_to<int32>();
  return vec_mul_rne_split(
      a, bi, ::aie::bit_and(int32(0xff), ::aie::downshift(bi, 23)),
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bi)));
}

// a * b as vec_mul_rne rounds it: the scalar soft-float `*`'s product, but 0
// below FLT_MIN, without __mulsf3 (the target has no scalar float
// multiplier). 8 lanes: the narrowest width AIE2P legalizes an integer
// bit_and/bit_or at.
__attribute__((noinline)) static float scalar_mul_rne(float a, float b) {
  return vec_mul_rne(::aie::broadcast<float, 8>(a),
                     ::aie::broadcast<float, 8>(b))
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

// vec_round_to_i8 of 32 fp32 lanes (as bits), in integers. For |v| >= 0.5
// the vadd.f of +-0.5 is exact (or rounds within an integer), so the result
// is sign * floor(|v| + 0.5) = sign * floor((m * 2^(E - 124) + 2^25) / 2^26)
// for v's significand m and exponent field E; below 0.5 it is 0, but for
// 0x3effffff, whose vadd.f rounds up to 1. The power of two is a bf16's
// integer (E clamped to 134: larger |v| saturate either way), and the floor
// one srs: (2^26 k + 2^25 + j) / 2^26 for 0 <= j < 2^26 rounds half to even
// to k, so sign * (m * 2^(E - 124) + 1) shifted by 26 is the result.
// Needs rounding_mode::conv_even and saturation_mode::saturate.
static inline ::aie::vector<int8, 32>
vec_round_to_i8_bits(::aie::vector<int32, 32> vb) {
  Sig24 m = sig24_planes(vb);
  const ::aie::vector<int16, 32> hi = m.hi;
  // See x_planes.
  const ::aie::vector<int32, 16> hv = hi.cast_to<int32>();
  m.hi = ::aie::bit_or(int32(0x00800080), ::aie::bit_and(int32(0x007f007f), hv))
             .cast_to<int16>();
  // bf16 2^(E - 124): exponent field E + 3.
  const ::aie::vector<int16, 32> pw_bits = ::aie::add(
      ::aie::min(::aie::bit_and(int32(0x7f807f80), hv).cast_to<int16>(),
                 int16(134 << 7)),
      int16(3 << 7));
  const ::aie::vector<int16, 32> pw(::shuffle(
      ::bfloat16_to_int(pw_bits.extract<16>(0).cast_to<bfloat16>(), 0),
      ::bfloat16_to_int(pw_bits.extract<16>(1).cast_to<bfloat16>(), 0), 2));
  const auto neg = ::aie::lt(hi, int16(0));
  const ::aie::vector<int16, 32> spw = ::aie::select(pw, ::aie::neg(pw), neg);
  const ::aie::vector<int16, 32> sgn = ::aie::select(int16(1), int16(-1), neg);
  v32acc64 p = ::mul_elem_32_conf(m.hi, 0, spw, 1, 0);
  p = ::mac_elem_32_conf(m.lo, 0, spw, 1, p, 0, 1, 0, 0);
  p = ::mac_elem_32_conf(sgn, 1, ::aie::broadcast<int16, 32>(1), 1, p, 0, 0, 0,
                         0);
  const ::aie::vector<int8, 32> n =
      ::aie::accum<acc64, 32>(p).template to_vector<int8>(26);
  return ::aie::select(
      n, sgn.template pack<int8>(),
      ::aie::eq(::aie::bit_and(int32(0x7fffffff), vb), int32(0x3effffff)));
}

// A positive normal fp32 factor b as vec_requant_i8 takes it: its
// significand planes, its sign and exponent bits less 126 << 23, and the
// largest x exponent field whose product with b is below 0.25 (so rounds to
// 0 whatever its exact value).
struct RequantFactor {
  Sig24 sig;
  int32 e;
  int32 x_small;
};

static inline RequantFactor requant_factor(float b) {
  const int32_t bits = __builtin_bit_cast(int32_t, b);
  return {
      sig24_planes(::aie::broadcast<int32, 32>((bits & 0x7fffff) | 0x800000)),
      (bits & 0x7f800000) - (126 << 23), (250 << 23) - (bits & 0x7f800000)};
}

// vec_round_to_i8(vec_mul_rne(x, b)) for 32 fp32 lanes (as bits) with
// |x * b| < 2^15: lanes whose product is below 0.25 (x zero or denormal
// among them) give 0, the others' products are normal, so one exact product
// of significands rounded once by srs (by 23, or 24 where it reaches 2^47).
// Needs rounding_mode::conv_even and saturation_mode::saturate.
static inline ::aie::vector<int8, 32>
vec_requant_i8(::aie::vector<int32, 32> xb, const RequantFactor &b) {
  ::aie::accum<acc64, 32> init;
  init.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 24);
  Sig24 x = sig24_planes(xb);
  x.hi = ::aie::bit_or(
             int32(0x00800080),
             ::aie::bit_and(
                 int32(0x007f007f),
                 ::aie::vector<int16, 32>(x.hi).template cast_to<int32>()))
             .template cast_to<int16>();
  ::aie::accum<acc64, 32> p(sig24_mac<true>(x, b.sig, init));
  ::aie::vector<int32, 32> lo = p.template to_vector<int32>(23);
  ::aie::vector<int32, 32> hi = p.template to_vector<int32>(24);
  ::aie::vector<int32, 32> v =
      ::aie::add(::aie::add(::aie::bit_and(int32(0xff800000), xb), b.e),
                 ::aie::select(lo, hi, ::aie::ge(lo, int32(0))));
  return ::aie::select(
      vec_round_to_i8_bits(v), ::aie::zeros<int8, 32>(),
      ::aie::le(::aie::bit_and(int32(0x7f800000), xb), b.x_small));
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

// recip_rne's bits for 32 lanes of positive normal s (as bits) with
// exponent fields in [3, 251], given r0 (bits) within an ulp of 1 / s, as
// aie::inv's are: 1 / s = (2^47 / m) * 2^(103 - E) for s's significand m and
// exponent field E, and q = round(2^47 / m) is r0's significand (scaled to
// 2^47 / m) or one off it: q0 - 1, q0 or q0 + 1 as 2 * (q0 * m - 2^47)
// exceeds m, lies within [-m, m] or falls below -m (never equal: m divides no
// 2^48 as an odd multiple). Needs rounding_mode::conv_even.
static inline ::aie::vector<int32, 32>
vec_recip_rne_bits(::aie::vector<int32, 32> sb, ::aie::vector<int32, 32> r0) {
  const ::aie::vector<int32, 32> se = ::aie::bit_and(int32(0x7f800000), sb);
  const ::aie::vector<int32, 32> r0m =
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), r0));
  // r0 in the binade above 2^23 / m's means q0 = 2 * r0's significand.
  const ::aie::vector<int32, 32> q0 = ::aie::select(
      r0m, ::aie::add(r0m, r0m),
      ::aie::eq(::aie::add(::aie::bit_and(int32(0x7f800000), r0), se),
                int32(254 << 23)));
  const ::aie::vector<int32, 32> m =
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), sb));
  ::aie::accum<acc64, 32> init;
  init.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 24);
  // q0 * m - 2^47, exactly (|.| < 2^25).
  const ::aie::vector<int32, 32> d =
      ::aie::accum<acc64, 32>(
          sig24_mac<true>(sig24_planes(q0), sig24_planes(m), init))
          .template to_vector<int32>(0);
  const ::aie::vector<int32, 32> d2 = ::aie::add(d, d);
  const ::aie::vector<int32, 32> q = ::aie::select(
      ::aie::select(q0, ::aie::sub(q0, int32(1)), ::aie::gt(d2, m)),
      ::aie::add(q0, int32(1)), ::aie::lt(d2, ::aie::neg(m)));
  // The significand's implicit bit carries into the exponent field, which
  // q = 2^24 (s a power of two) bumps once more.
  return ::aie::add(::aie::sub(int32(252 << 23), se), q);
}

// recip_rne lane-wise: the same restoring division of the significand, on
// kN lanes at once.
template <unsigned kN>
__attribute__((noinline)) static ::aie::vector<float, kN>
vec_recip_rne(::aie::vector<float, kN> a) {
  ::aie::vector<int32, kN> bits = a.template cast_to<int32>();
  ::aie::vector<int32, kN> e =
      ::aie::downshift(::aie::bit_and(int32(0x7f800000), bits), 23);
  ::aie::vector<int32, kN> m =
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bits));
  ::aie::vector<int32, kN> rem = ::aie::broadcast<int32, kN>(1 << 24);
  ::aie::vector<int32, kN> q = ::aie::zeros<int32, kN>();
  for (int i = 0; i < 24; i++) {
    auto ge = ::aie::ge(rem, m);
    q = ::aie::add(::aie::add(q, q),
                   ::aie::select(::aie::zeros<int32, kN>(),
                                 ::aie::broadcast<int32, kN>(1), ge));
    rem = ::aie::select(rem, ::aie::sub(rem, m), ge);
    rem = ::aie::add(rem, rem);
  }
  auto exact = ::aie::eq(q, int32(1 << 23)) & ::aie::eq(rem, int32(0));
  auto up =
      ::aie::gt(rem, m) |
      (::aie::eq(rem, m) & ::aie::gt(::aie::bit_and(int32(1), q), int32(0)));
  q = ::aie::select(q, ::aie::add(q, int32(1)), up);
  return ::aie::select(::aie::add(::aie::upshift(::aie::sub(int32(253), e), 23),
                                  ::aie::sub(q, int32(1 << 23))),
                       ::aie::upshift(::aie::sub(int32(254), e), 23), exact)
      .template cast_to<float>();
}

// 1e-12f's bits. A non-NaN float's |v| bits order as an int32, so an
// absmax's floor is an integer max.
static constexpr int32_t kAbsmaxFloorBits = 0x2b8cbccc;

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
  // Every head's scale and reciprocal in one group of kV lanes: AIE2P
  // legalizes vec_recip_rne's integer shifts at no fewer.
  constexpr int kGroup = kV;
  static_assert(kHeads <= kGroup);
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
  ::aie::vector<float, kGroup> absmax =
      ::aie::max(::aie::load_v<kGroup>(q_scale).template cast_to<int32>(),
                 kAbsmaxFloorBits)
          .template cast_to<float>();
  ::aie::vector<float, kGroup> scale =
      vec_mul_rne(absmax, ::aie::broadcast<float, kGroup>(1.0f / 127.0f));
  ::aie::store_v(q_scale, scale);
  ::aie::store_v(q_inv, vec_recip_rne(scale));
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

#endif // LLAMA_VEC_FP_H
