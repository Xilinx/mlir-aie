//===- log_f32.h ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_LOG_F32_H
#define AIE_KERNELS_COMMON_LOG_F32_H

#include "../aie_arch.h"
#include "../aie_kernel_utils.h"
#include "f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// log(m) for m in [sqrt(1/2), sqrt(2)) is f q(f) with f = m - 1, and q is
// log1p(f) / f interpolated at the 10 Chebyshev nodes of that range (numpy
// polyfit, degree 9): 2.5e-9 from log1p(f) with these float32 coefficients.
static const float log_q[10] = {1.0f,
                                -0.49999988079071045f,
                                0.33333346247673035f,
                                -0.2500157952308655f,
                                0.20000939071178436f,
                                -0.1660836935043335f,
                                0.14199650287628174f,
                                -0.13266031444072723f,
                                0.12806610763072968f,
                                -0.07451186329126358f};

// The limbs of f and of q(f), f q(f) approximating log1p(f) for f in
// [sqrt(1/2) - 1, sqrt(2) - 1).
struct LogPoly {
  bf16_vec fh, fm, fl, qh, qm, ql;
};

static inline __attribute__((always_inline)) LogPoly log_poly(f32_acc f,
                                                              bf16_vec one) {
  LogPoly lp;
  split3(f, lp.fh, lp.fm, lp.fl, one);
  f32_acc q =
      aie::add(f32_acc(aie::broadcast<float, f32_lanes>(0.0f)), log_q[9]);
  for (int k = 8; k >= 0; k--) {
    bf16_vec qh, qm, ql;
    split3(q, qh, qm, ql, one);
    q = aie::mul(ql, lp.fh);
    q = aie::mac(aie::mac(q, qm, lp.fm), qh, lp.fl);
    q = aie::mac(aie::mac(q, qm, lp.fh), qh, lp.fm);
    q = aie::add(aie::mac(q, qh, lp.fh), log_q[k]);
  }
  split3(q, lp.qh, lp.qm, lp.ql, one);
  return lp;
}

// r + f q(f), smallest products first.
static inline __attribute__((always_inline)) f32_acc
log_poly_add(f32_acc r, const LogPoly &lp) {
  r = aie::mac(aie::mac(aie::mac(r, lp.ql, lp.fh), lp.qm, lp.fm), lp.qh, lp.fl);
  return aie::mac(aie::mac(aie::mac(r, lp.qm, lp.fh), lp.qh, lp.fm), lp.qh,
                  lp.fh);
}

// log(z) of 32 float32 lanes, z a positive normal float32 (in an
// accumulator), in an f32 accumulator: log_f32.cc's arithmetic, for a kernel
// that keeps the result in float32. Multiplies as f32_split.h does, under the
// caller's conv_even rounding mode.
static inline __attribute__((always_inline)) f32_acc log_f32_acc(f32_acc z) {
  const bf16_vec one = bf16_splat(0x3f80);
  auto bits = z.to_vector<float>().cast_to<int32_t>();
  // z = 2^e m with m in [sqrt(1/2), sqrt(2)); downshift floors in any mode.
  auto e = aie::downshift(
      aie::sub(bits, aie::broadcast<int32_t, f32_lanes>(0x3f3504f3)), 23);
  auto m = aie::sub(bits, aie::upshift(e, 23)).cast_to<float>();
  // e as a float: 1.5 * 2^23 + e has e in its low mantissa bits.
  auto e_magic = aie::add(e, aie::broadcast<int32_t, f32_lanes>(0x4b400000));
  bf16_vec ef = aie::add(f32_acc(e_magic.cast_to<float>()), -12582912.0f)
                    .to_vector<bfloat16>();
#if AIE_ARCH_AIE2
  // As in split3: m - 1 in halves, so m < 1 is not rounded to 1's exponent.
  const bf16_vec half = bf16_splat(0x3f00);
  f32_acc f = aie::msc(aie::msc(f32_acc(m), one, half), one, half);
#else
  f32_acc f = aie::msc(f32_acc(m), one, one);
#endif
  const LogPoly lp = log_poly(f, one);
  // ln 2 as three bf16 limbs.
  f32_acc r = aie::mul(ef, bf16_splat(0x35c0));
  r = aie::mac(r, ef, bf16_splat(0x3ae4));
  r = log_poly_add(r, lp);
  return aie::mac(r, ef, bf16_splat(0x3f31));
}

#endif
