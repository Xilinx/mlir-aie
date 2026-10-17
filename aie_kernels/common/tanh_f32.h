//===- tanh_f32.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_TANH_F32_H
#define AIE_KERNELS_COMMON_TANH_F32_H

#include "exp2_bf16.h"
#include "f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// k0 tanh(s / c) for f32 s, as k0 (2r - 1) with r = 1 / (1 + 2^a) and
// a = -|s| * 2 log2(e) / c <= 0; 2^a flushing to 0 saturates it at k0.
// a_limbs is a's factor 2 log2(e) / c (times any scale s takes first),
// k1_limbs 2 k0. With k0 = c log2(e) it is a soft cap in the log2 domain.
static inline __attribute__((always_inline)) aie::accum<accfloat, f32_lanes>
scaled_tanh_f32(aie::accum<accfloat, f32_lanes> s,
                const Limbs<f32_lanes> &a_limbs, float k0,
                const Limbs<f32_lanes> &k1_limbs) {
  using Acc = aie::accum<accfloat, f32_lanes>;
  auto splat = [](float c) { return Acc(aie::broadcast<float, f32_lanes>(c)); };
  auto limb = [](int16_t bits) {
    return aie::broadcast<int16_t, f32_lanes>(bits).cast_to<bfloat16>();
  };
  // The sign by the bits' compare and -s in the accumulator: Peano cannot
  // legalize a 32-lane f32 fneg or 1024-bit OR.
  const auto sv = s.to_vector<float>();
  const auto negative =
      aie::lt(sv.cast_to<int32_t>(), aie::zeros<int32_t, f32_lanes>());
  const auto neg = aie::sub(aie::zeros<accfloat, f32_lanes>(), sv);
  const Acc neg_abs(aie::select(neg.to_vector<float>(), sv, negative));
  const aie::vector<float, f32_lanes> a =
      mac_limbs(aie::zeros<accfloat, f32_lanes>(), bf16_limbs(neg_abs), a_limbs)
          .to_vector<float>();
  const Exp2F32 e2a = exp2_f32_32(a);
  const Acc d = aie::add(
      splat(1.0f), aie::select(e2a.bits.cast_to<float>(),
                               aie::zeros<float, f32_lanes>(), e2a.below));

  // r = 1 / d, d in [1, 2]: a quadratic within 2%, then two Newton steps
  // r -= r (d r - 1), the first from r in bf16.
  const Limbs<f32_lanes> dl = bf16_limbs(d);
  const auto q =
      aie::mac(splat(-1.4515766f), dl.hi, limb(0x3ea5)).to_vector<bfloat16>();
  const auto r0 = aie::mac(splat(2.1190739f), q, dl.hi).to_vector<bfloat16>();
  Limbs<f32_lanes> e = bf16_limbs(aie::mac(
      aie::mac(aie::mac(splat(-1.0f), dl.hi, r0), dl.mid, r0), dl.lo, r0));
  const Acc r1 = aie::msc(aie::msc(Acc(r0), r0, e.hi), r0, e.mid);
  const Limbs<f32_lanes> rl = bf16_limbs(r1);
  e = bf16_limbs(mac_limbs(splat(-1.0f), dl, rl));
  const Acc r =
      aie::msc(aie::msc(aie::msc(r1, rl.hi, e.hi), rl.hi, e.mid), rl.mid, e.hi);

  const auto t = mac_limbs(splat(-k0), bf16_limbs(r), k1_limbs);
  return Acc(aie::select(
      t.to_vector<float>(),
      aie::sub(aie::zeros<accfloat, f32_lanes>(), t.to_vector<float>())
          .to_vector<float>(),
      negative));
}

// sigmoid(x) = 1 / (1 + 2^(-x log2(e))) for f32 x, within about 2e-5 of it,
// relatively, over the whole range: exp2_f32_32's 2^x, then 1 / d from d's
// mantissa in [1, 2) (inv_1_2) and its exponent. Below the smallest normal it
// flushes to 0; NaN passes through. x is held at +-200, where sigmoid is
// already 1 or 0, so infinities split into finite limbs.
static inline __attribute__((always_inline)) aie::accum<accfloat, f32_lanes>
sigmoid_f32(aie::accum<accfloat, f32_lanes> x) {
  using Acc = aie::accum<accfloat, f32_lanes>;
  using Int = aie::vector<int32_t, f32_lanes>;
  auto splat = [](float c) { return Acc(aie::broadcast<float, f32_lanes>(c)); };
  const auto xi = x.to_vector<float>().cast_to<int32_t>();
  const Int abs_bits =
      aie::bit_and(xi, aie::broadcast<int32_t, f32_lanes>(0x7fffffff));
  const auto hold =
      aie::gt(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x43480000)) &
      aie::le(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x7f800000));
  const Int bound = aie::select(
      aie::broadcast<int32_t, f32_lanes>(0x43480000),
      aie::broadcast<int32_t, f32_lanes>(static_cast<int32_t>(0xc3480000u)),
      aie::lt(xi, aie::zeros<int32_t, f32_lanes>()));
  const Acc held(aie::select(xi, bound, hold).template cast_to<float>());
  const Limbs<f32_lanes> neg_log2e = bf16_limbs(splat(-1.44269504088896341f));
  const Exp2F32 e = exp2_f32_32(
      mac_limbs(aie::zeros<accfloat, f32_lanes>(), bf16_limbs(held), neg_log2e)
          .to_vector<float>());
  const Acc d = aie::add(splat(1.0f),
                         aie::select(e.bits.cast_to<float>(),
                                     aie::zeros<float, f32_lanes>(), e.below));
  const Int dbits = d.to_vector<float>().cast_to<int32_t>();
  const Int k = aie::bit_and(
      aie::sub(dbits, aie::broadcast<int32_t, f32_lanes>(0x3f800000)),
      aie::broadcast<int32_t, f32_lanes>(0x7f800000));
  const Int r =
      aie::sub(inv_1_2(Acc(aie::sub(dbits, k).template cast_to<float>()))
                   .to_vector<float>()
                   .cast_to<int32_t>(),
               k);
  const auto flush =
      e.overflow | aie::ge(k, aie::broadcast<int32_t, f32_lanes>(126 << 23));
  const auto out =
      aie::select(r.cast_to<float>(), aie::zeros<float, f32_lanes>(), flush);
  return Acc(aie::select(out, x.to_vector<float>(), e.is_nan));
}

// tanh(t) for f32 t, within about 3e-5 of it, relatively: below |t| = 1/16
// its odd series to t^7, else sign(t) (1 - 2 sigmoid(-2 |t|)), where the
// subtraction keeps 1 - 2 sigmoid above 1/16 of its operands. |t| is held at
// 16, where tanh is already 1 in f32; NaN passes through.
static inline __attribute__((always_inline)) aie::accum<accfloat, f32_lanes>
tanh_f32(aie::accum<accfloat, f32_lanes> t) {
  using Acc = aie::accum<accfloat, f32_lanes>;
  auto splat = [](float c) { return Acc(aie::broadcast<float, f32_lanes>(c)); };
  const auto tv = t.to_vector<float>();
  const auto ti = tv.cast_to<int32_t>();
  const auto negative = aie::lt(ti, aie::zeros<int32_t, f32_lanes>());
  const auto abs_bits =
      aie::bit_and(ti, aie::broadcast<int32_t, f32_lanes>(0x7fffffff));
  const auto small =
      aie::lt(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x3d800000));
  const auto hold =
      aie::gt(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x41800000)) &
      aie::le(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x7f800000));
  const auto abs =
      aie::select(abs_bits, aie::broadcast<int32_t, f32_lanes>(0x41800000),
                  hold)
          .cast_to<float>();
  const Limbs<f32_lanes> al = bf16_limbs(Acc(abs));

  const Limbs<f32_lanes> two = bf16_limbs(splat(-2.0f));
  const Acc s =
      sigmoid_f32(mac_limbs(aie::zeros<accfloat, f32_lanes>(), al, two));
  const Acc big = mac_limbs(splat(1.0f), bf16_limbs(s), two);

  const Limbs<f32_lanes> u =
      bf16_limbs(mac_limbs(aie::zeros<accfloat, f32_lanes>(), al, al));
  Acc q = splat(-17.0f / 315.0f);
  q = mac_limbs(splat(2.0f / 15.0f), u, bf16_limbs(q));
  q = mac_limbs(splat(-1.0f / 3.0f), u, bf16_limbs(q));
  q = mac_limbs(splat(1.0f), u, bf16_limbs(q));
  const Acc series =
      mac_limbs(aie::zeros<accfloat, f32_lanes>(), al, bf16_limbs(q));

  const auto mag =
      aie::select(big.to_vector<float>(), series.to_vector<float>(), small);
  const auto neg =
      aie::sub(aie::zeros<accfloat, f32_lanes>(), mag).to_vector<float>();
  return Acc(aie::select(mag, neg, negative));
}

#endif
