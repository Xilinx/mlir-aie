// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_EXP2_BF16_H
#define AIE_KERNELS_COMMON_EXP2_BF16_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// 2^x in f32 for every x: 2^k * p(f), k = round(x), with p a polynomial in
// f = x - k (|f| <= 1/2) evaluated by Horner's rule on bf16 limbs, within
// 2.8e-6 of 2^f. Lanes past f32's range (x < -125.5, x >= 128) and NaN are
// flagged for the caller. The caller sets conv_even.
struct Exp2F32 {
  aie::vector<int32_t, 32> bits;
  aie::mask<32> is_nan, below, overflow;
};

static inline __attribute__((always_inline)) Exp2F32
exp2_f32_32(aie::vector<float, 32> x) {
  using acc_t = aie::accum<accfloat, 32>;
  using bf_t = aie::vector<bfloat16, 32>;
  auto bf = [](int16_t bits) {
    return aie::broadcast<int16_t, 32>(bits).cast_to<bfloat16>();
  };
  auto acc = [](float c) { return acc_t(aie::broadcast<float, 32>(c)); };
  auto bits = [](float v) {
    return aie::broadcast<float, 32>(v).cast_to<int32_t>();
  };
  // The f32 compares are emulated, so the bits are compared: they order as
  // signed integers among positive values and in reverse as unsigned ones
  // among negative values. Doubling them drops the sign, leaving NaNs above
  // 0xff000000.
  const auto xi = x.cast_to<int32_t>();
  const auto xu = xi.cast_to<uint32_t>();
  const auto is_nan =
      aie::gt(aie::add(xu, xu), aie::broadcast<uint32_t, 32>(0xff000000u));
  const auto below = aie::gt(xu, bits(-125.5f).cast_to<uint32_t>());
  const auto overflow = aie::ge(xi, bits(128.0f));

  // The 1.5 * 2^23 rounding: see exp2_poly.h. x - k goes through the
  // accumulator, where k is exact in bf16.
  const auto magic = aie::broadcast<float, 32>(12582912.0f);
  const aie::vector<float, 32> xm = aie::add(x, magic);
  const bf_t one = bf(0x3f80);
  const bf_t k = acc_t(aie::sub(xm, magic)).to_vector<bfloat16>();
  const acc_t f = aie::msc(acc_t(x), k, one);
  const bf_t f_hi = f.to_vector<bfloat16>();
  const bf_t f_lo = aie::msc(f, f_hi, one).to_vector<bfloat16>();

  acc_t p = aie::mac(acc(0.0559063815f), f_hi, bf(0x3c1d));
  auto step = [&](float c) {
    const bf_t hi = p.to_vector<bfloat16>();
    const bf_t lo = aie::msc(p, hi, one).to_vector<bfloat16>();
    acc_t q = aie::mac(acc(c), hi, f_hi);
    q = aie::mac(q, hi, f_lo);
    p = aie::mac(q, lo, f_hi);
  };
  step(0.240241051f);
  step(0.693124175f);
  step(1.0f);

  // k << 23 is bits(xm) << 23.
  const auto y = aie::add(p.to_vector<float>().cast_to<int32_t>(),
                          aie::upshift(xm.cast_to<int32_t>(), 23));
  return {y, is_nan, below, overflow};
}

// 2^x rounded to bf16: a polynomial under -DEXP2_BF16_ACCURATE (below) on
// either architecture, else aie::exp2<bfloat16> on AIE2P. AIE2 has no exp2, so
// there x = k + f with k = round(x) and |f| <= 1/2. 2^f is a cubic in bf16
// products, within 0.2% of 2^f, and 2^k is added into its f32 exponent field.
// x is clamped to [-200, 127]. Past the bottom of the exponent field the sum
// goes negative, and is clamped to +0.
#if !AIE_HAS_NATIVE_EXP2 && !defined(EXP2_BF16_ACCURATE)
static inline aie::vector<bfloat16, 16> exp2_bf16_16(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-200.0f));
  x = aie::min(x, aie::broadcast<float, 16>(127.0f));
  // The 1.5 * 2^23 rounding: see exp2_poly.h.
  const auto magic = aie::broadcast<float, 16>(12582912.0f);
  const aie::vector<float, 16> xm = aie::add(x, magic);
  const aie::vector<int32_t, 16> k =
      aie::sub(xm.cast_to<int32_t>(), aie::broadcast<int32_t, 16>(0x4b400000));
  aie::accum<accfloat, 16> fa;
  fa.from_vector(aie::sub(x, aie::sub(xm, magic)));
  const aie::vector<bfloat16, 16> f = fa.to_vector<bfloat16>();

  auto horner = [&](aie::vector<bfloat16, 16> c1, float c0) {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::broadcast<float, 16>(c0));
    return aie::mac(acc, f, c1);
  };
  auto t = horner(aie::broadcast<bfloat16, 16>(0.0555041087f), 0.2402265069f);
  t = horner(t.to_vector<bfloat16>(), 0.6931471805f);
  t = horner(t.to_vector<bfloat16>(), 1.0f);

  aie::accum<accfloat, 16> r;
  r.from_vector(aie::max(aie::add(t.to_vector<float>().cast_to<int32_t>(),
                                  aie::upshift(k, 23)),
                         aie::zeros<int32_t, 16>())
                    .cast_to<float>());
  return r.to_vector<bfloat16>();
}

template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  if constexpr (N < 16) {
    return exp2_bf16_16(x.template grow<16>()).template extract<N>(0);
  } else {
    aie::vector<bfloat16, N> out;
    for (unsigned i = 0; i < N / 16; i++)
      out.insert(i, exp2_bf16_16(x.template extract<16>(i)));
    return out;
  }
}
#elif defined(EXP2_BF16_ACCURATE)
// exp2_f32_32 rounded once to bf16 (aie::exp2<bfloat16> is off by up to 49%):
// x < -125.5 gives +0, x >= 128 +inf. The caller sets conv_even.
static inline aie::vector<bfloat16, 32> exp2_bf16_32(aie::vector<float, 32> x) {
  using acc_t = aie::accum<accfloat, 32>;
  using bf_t = aie::vector<bfloat16, 32>;
  auto bf = [](int16_t bits) {
    return aie::broadcast<int16_t, 32>(bits).cast_to<bfloat16>();
  };
  const Exp2F32 e = exp2_f32_32(x);
  bf_t r = acc_t(e.bits.cast_to<float>()).to_vector<bfloat16>();
  r = aie::select(r, aie::zeros<bfloat16, 32>(), e.below);
  r = aie::select(r, bf(0x7f80), e.overflow);
  return aie::select(r, bf(0x7fc0), e.is_nan);
}

template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  if constexpr (N < 32) {
    return exp2_bf16_32(x.template grow<32>()).template extract<N>(0);
  } else {
    aie::vector<bfloat16, N> out;
    for (unsigned i = 0; i < N / 32; i++)
      out.insert(i, exp2_bf16_32(x.template extract<32>(i)));
    return out;
  }
}

#define EXP2_SPLAT64(v)                                                        \
  {v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,           \
   v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,           \
   v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v}

// exp2_f32_32's constants, a vector each, loaded where they are used: held,
// they take the accumulator registers the chain needs.
alignas(64) static const float exp2_magic[64] = EXP2_SPLAT64(12582912.0f);
alignas(64) static const float exp2_c3[64] = EXP2_SPLAT64(0.0559063815f);
alignas(64) static const float exp2_c2[64] = EXP2_SPLAT64(0.240241051f);
alignas(64) static const float exp2_c1[64] = EXP2_SPLAT64(0.693124175f);
alignas(64) static const float exp2_c0[64] = EXP2_SPLAT64(1.0f);
// -125.5's bits: an x <= 0 is below it when its bits are above them.
alignas(64) static const uint32_t exp2_floor[64] = EXP2_SPLAT64(0xc2fb0000u);
alignas(64) static const uint16_t exp2_one[64] = EXP2_SPLAT64(0x3f80);
alignas(64) static const uint16_t exp2_c4[64] = EXP2_SPLAT64(0x3c1d);

// The lanes exp2_nonpos_bf16 takes at once where a row allows: AIE2P's
// 64-lane accumulators.
#if AIE_ARCH_AIE2P
constexpr unsigned EXP2_NONPOS_LANES = 64;
#else
constexpr unsigned EXP2_NONPOS_LANES = 32;
#endif

// exp2_bf16_32 of x <= 0, finite or -inf, as the online softmax takes it: no
// such x reaches its overflow or NaN select, so only the flush below -125.5
// stays. exp2_f32_32's operations in its order, lane for lane, at L lanes;
// k << 23 is taken as soon as k is, and the flush applied to the f32 bits,
// so the one rounding to bf16 is the store's.
template <unsigned L>
static inline __attribute__((always_inline)) aie::vector<bfloat16, L>
exp2_nonpos_lanes(aie::vector<float, L> x) {
  using acc_t = aie::accum<accfloat, L>;
  using bf_t = aie::vector<bfloat16, L>;
  auto ld = [](const float *c) { return aie::load_v<L>(c); };
  auto bf = [](const uint16_t *c) {
    return aie::load_v<L>(c).template cast_to<bfloat16>();
  };
  const auto below =
      aie::gt(x.template cast_to<uint32_t>(), aie::load_v<L>(exp2_floor));
  const aie::vector<float, L> xm = aie::add(x, ld(exp2_magic));
  const auto k23 = aie::upshift(xm.template cast_to<int32_t>(), 23);
  const bf_t k =
      acc_t(aie::sub(xm, ld(exp2_magic))).template to_vector<bfloat16>();
  const acc_t f = aie::msc(acc_t(x), k, bf(exp2_one));
  const bf_t f_hi = f.template to_vector<bfloat16>();
  const bf_t f_lo =
      aie::msc(f, f_hi, bf(exp2_one)).template to_vector<bfloat16>();

  acc_t p = aie::mac(acc_t(ld(exp2_c3)), f_hi, bf(exp2_c4));
  auto step = [&](const float *c) {
    const bf_t hi = p.template to_vector<bfloat16>();
    const bf_t lo =
        aie::msc(p, hi, bf(exp2_one)).template to_vector<bfloat16>();
    acc_t q = aie::mac(acc_t(ld(c)), hi, f_hi);
    q = aie::mac(q, hi, f_lo);
    p = aie::mac(q, lo, f_hi);
  };
  step(exp2_c2);
  step(exp2_c1);
  step(exp2_c0);

  const auto y =
      aie::add(p.template to_vector<float>().template cast_to<int32_t>(), k23);
  return acc_t(aie::select(y, aie::zeros<int32_t, L>(), below)
                   .template cast_to<float>())
      .template to_vector<bfloat16>();
}

template <unsigned N>
static inline aie::vector<bfloat16, N>
exp2_nonpos_bf16(aie::vector<float, N> x) {
  constexpr unsigned L = N % EXP2_NONPOS_LANES == 0 ? EXP2_NONPOS_LANES : 32;
  if constexpr (N < 32) {
    return exp2_nonpos_lanes<32>(x.template grow<32>()).template extract<N>(0);
  } else if constexpr (N == L) {
    return exp2_nonpos_lanes<L>(x);
  } else {
    aie::vector<bfloat16, N> out;
#pragma unroll
    for (unsigned i = 0; i < N / L; i++)
      out.insert(i, exp2_nonpos_lanes<L>(x.template extract<L>(i)));
    return out;
  }
}
#else
template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  return aie::exp2<bfloat16>(x);
}
#endif

// Without the accurate exp2, exp2_bf16 itself.
#ifndef EXP2_BF16_ACCURATE
constexpr unsigned EXP2_NONPOS_LANES = 32;

template <unsigned N>
static inline aie::vector<bfloat16, N>
exp2_nonpos_bf16(aie::vector<float, N> x) {
  return exp2_bf16(x);
}
#endif

#endif
