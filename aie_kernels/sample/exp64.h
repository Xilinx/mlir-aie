//===- exp64.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// exp over IEEE binary64 as a fixed sequence of IEEE +, -, * and integer bit
// operations (no fma, libm or division): under -ffp-contract=off every
// conforming target returns the same bits, which
// aie.iron.kernels.sample.exp64_ref reproduces. ARM optimized-routines' exp
// (a 128-entry table with rounding tails, a scaled subnormal path) with a
// Taylor polynomial and a 35-bit Cody-Waite high part, so kd * NEG_LN2_HI_N
// is exact without fma. One operation per statement, in exp64_ref's order.

#ifndef AIE_KERNELS_SAMPLE_EXP64_H
#define AIE_KERNELS_SAMPLE_EXP64_H

#include <stdint.h>

#include "exp64_table.h"

static_assert(sizeof(double) == 8, "exp64 needs IEEE binary64 doubles");

static inline double exp64_as_double(uint64_t u) {
  return __builtin_bit_cast(double, u);
}

static inline uint64_t exp64_as_bits(double d) {
  return __builtin_bit_cast(uint64_t, d);
}

// The top 12 bits (sign and exponent) of a double.
static inline uint32_t exp64_top12(double x) {
  return (uint32_t)(exp64_as_bits(x) >> 52);
}

// |x| >= 512: the scale's exponent field would over- or underflow.
static inline double exp64_special(double tmp, uint64_t sbits, uint64_t ki) {
  if ((ki & 0x80000000u) == 0) {
    // k > 0: scale by 2^1009 afterwards; overflows to +inf past ~709.78.
    sbits -= UINT64_C(1009) << 52;
    double scale = exp64_as_double(sbits);
    double st = scale * tmp;
    double y = scale + st;
    return 0x1p1009 * y;
  }
  // k < 0: scale by 2^-1022 afterwards. When the result is subnormal, round y
  // to the subnormal's precision first (by adding 1.0) so the final scaling
  // is exact, rather than rounding twice.
  sbits += UINT64_C(1022) << 52;
  double scale = exp64_as_double(sbits);
  double st = scale * tmp;
  double y = scale + st;
  if (y < 1.0) {
    double lo = scale - y;
    lo = lo + st;
    double hi = 1.0 + y;
    double lo2 = 1.0 - hi;
    lo2 = lo2 + y;
    lo2 = lo2 + lo;
    y = hi + lo2;
    y = y - 1.0;
    if (y == 0.0)
      y = 0.0; // never -0
  }
  return 0x1p-1022 * y;
}

static inline double exp64(double x) {
  uint32_t abstop = exp64_top12(x) & 0x7ff;
  // Out of [2^-54, 512): tiny, huge, inf, nan.
  if (abstop - exp64_top12(0x1p-54) >=
      exp64_top12(512.0) - exp64_top12(0x1p-54)) {
    if (abstop - exp64_top12(0x1p-54) >= 0x80000000u)
      return 1.0 + x; // |x| < 2^-54, zeros included: 1.0 or its neighbour
    if (abstop >= exp64_top12(1024.0)) {
      if (exp64_as_bits(x) == exp64_as_bits(-__builtin_inf()))
        return 0.0;
      if (abstop >= exp64_top12(__builtin_inf()))
        return 1.0 + x; // +inf -> +inf, nan -> nan
      if (exp64_as_bits(x) >> 63)
        return 0.0;
      return __builtin_inf();
    }
    abstop = 0; // 512 <= |x| < 1024: the special path below
  }

  double z = exp64_as_double(EXP64_INV_LN2_N) * x;
  // Round z to an integer with the 1.5*2^52 shift: k lands in the low bits.
  double kd = z + exp64_as_double(EXP64_SHIFT);
  uint64_t ki = exp64_as_bits(kd);
  kd = kd - exp64_as_double(EXP64_SHIFT);
  // r = x - k*ln2/N; kd * hi is exact (35-bit hi, |k| < 2^18).
  double khi = kd * exp64_as_double(EXP64_NEG_LN2_HI_N);
  double klo = kd * exp64_as_double(EXP64_NEG_LN2_LO_N);
  double r = x + khi;
  r = r + klo;
  // 2^(k/N) = scale * (1 + tail).
  uint64_t idx = 2 * (ki % EXP64_N);
  uint64_t top = ki << (52 - EXP64_TABLE_BITS);
  double tail = exp64_as_double(exp64_table[idx]);
  uint64_t sbits = exp64_table[idx + 1] + top;
  // tmp = tail + (exp(r) - 1) = tail + r + r^2*(C2 + r*C3) + r^4*(C4 + r*C5).
  double r2 = r * r;
  double p23 = r * exp64_as_double(EXP64_C3);
  p23 = exp64_as_double(EXP64_C2) + p23;
  p23 = r2 * p23;
  double p45 = r * exp64_as_double(EXP64_C5);
  p45 = exp64_as_double(EXP64_C4) + p45;
  double r4 = r2 * r2;
  p45 = r4 * p45;
  double tmp = tail + r;
  tmp = tmp + p23;
  tmp = tmp + p45;
  if (abstop == 0)
    return exp64_special(tmp, sbits, ki);
  double scale = exp64_as_double(sbits);
  double st = scale * tmp;
  return scale + st;
}

#endif
