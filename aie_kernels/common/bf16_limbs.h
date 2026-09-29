//===- bf16_limbs.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_BF16_LIMBS_H
#define AIE_KERNELS_COMMON_BF16_LIMBS_H

#include <aie_api/aie.hpp>
#include <stdint.h>

// f32 arithmetic on AIE2 without its soft-float scalar unit or emulated f32
// vector multiply: an f32 is split into bf16 limbs, and a bf16 mac sums two
// exact products into each f32 lane (lane i gets a[i] b[i] + a[i+16] b[i+16]),
// so an f32 held as [hi | lo] limbs times a bf16 is one mac.
static inline v32bfloat16 bf16_pair(v16bfloat16 lo, v16bfloat16 hi) {
  return concat(lo, hi);
}

static inline v16bfloat16 bf16_lanes(int32_t bits) {
  return extract_v16bfloat16(
      broadcast_to_v32bfloat16(__builtin_bit_cast(bfloat16, (int16_t)bits)), 0);
}

// x - hi, exactly.
static inline v16accfloat residual(v16accfloat x, v16bfloat16 hi) {
  return msc_elem_16_2(bf16_pair(hi, bf16_lanes(0)),
                       broadcast_one_to_v32bfloat16(), x);
}

// [hi | mid] limbs of x, holding its top 16 bits; rest is x - hi - mid,
// exactly, in one msc.
static inline v32bfloat16 limbs(v16accfloat x, v16accfloat &rest) {
  v16bfloat16 hi = to_v16bfloat16(x);
  v32bfloat16 l = bf16_pair(hi, to_v16bfloat16(residual(x, hi)));
  rest = msc_elem_16_2(l, broadcast_one_to_v32bfloat16(), x);
  return l;
}

static inline v32bfloat16 limbs(v16accfloat x) {
  v16accfloat rest;
  return limbs(x, rest);
}

static inline v16bfloat16 limb(v32bfloat16 x, int i) {
  return extract_v16bfloat16(x, i);
}

// 1 / sqrt(m) as [hi | lo] limbs, to about 1e-5, for a normal m > 0; m holds
// m_acc's limbs.
static inline v32bfloat16 inv_sqrt_limbs(v16accfloat m_acc, v32bfloat16 m) {
  // y0: the bit-trick estimate, cut to bf16 (4% off).
  float m_f = ::aie::vector<float, 16>(v16float(m_acc))[0];
  int32_t y0_bits =
      (0x5f3759df - (__builtin_bit_cast(int32_t, m_f) >> 1)) >> 16;
  v16bfloat16 y0 = bf16_lanes(y0_bits);
  v16bfloat16 zero = bf16_lanes(0);

  // y1 = y0 + y0 / 2 (1 - m y0^2), a Newton step in bf16 (5e-3 off).
  v16bfloat16 m_y0 = to_v16bfloat16(mul_elem_16_2(m, bf16_pair(y0, y0)));
  v16accfloat r = msc_elem_16_2(bf16_pair(m_y0, zero), bf16_pair(y0, zero),
                                broadcast_to_v16accfloat(1.0f));
  v16bfloat16 y1 = to_v16bfloat16(mac_elem_16_2(
      bf16_pair(to_v16bfloat16(r), zero),
      bf16_pair(bf16_lanes(y0_bits - 0x80), zero), ups_to_v16accfloat(y0)));

  // y = y1 (1 + r / 2 + 3 r^2 / 8) with r = 1 - m y1^2 taken exactly: y1^2 is
  // exact in two limbs, and |r| < 1e-2 leaves the cubic term under 4e-7.
  bfloat16 y1_s = ::aie::vector<bfloat16, 16>(y1)[0];
  int32_t y1_bits = __builtin_bit_cast(int16_t, y1_s);
  v32bfloat16 y1_sq =
      limbs(mul_elem_16_2(bf16_pair(y1, zero), bf16_pair(y1, zero)));
  r = msc_elem_16_2(m, bf16_pair(limb(y1_sq, 1), limb(y1_sq, 1)),
                    broadcast_to_v16accfloat(1.0f));
  r = msc_elem_16_2(m, bf16_pair(limb(y1_sq, 0), limb(y1_sq, 0)), r);
  v32bfloat16 r_limbs = limbs(r);
  v16bfloat16 r_sq = to_v16bfloat16(mul_elem_16_2(
      bf16_pair(limb(r_limbs, 0), zero), bf16_pair(limb(r_limbs, 0), zero)));
  v16bfloat16 y1_3_8 = to_v16bfloat16(mul_elem_16_2(
      bf16_pair(y1, zero), bf16_pair(bf16_lanes(0x3ec0), zero))); // 0.375
  v16bfloat16 y1_half = bf16_lanes(y1_bits - 0x80);
  v16accfloat y = mac_elem_16_2(r_limbs, bf16_pair(y1_half, y1_half),
                                ups_to_v16accfloat(y1));
  y = mac_elem_16_2(bf16_pair(r_sq, zero), bf16_pair(y1_3_8, zero), y);
  return limbs(y);
}

static inline v32bfloat16 inv_sqrt_limbs(v16accfloat m_acc) {
  return inv_sqrt_limbs(m_acc, limbs(m_acc));
}

#endif
