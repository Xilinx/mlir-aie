// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_SCALAR_F32_H
#define AIE_KERNELS_COMMON_SCALAR_F32_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>

// AIE2P has no scalar float multiply, divide or C-style int-to-float convert;
// those lower to the soft-float helpers __mulsf3, __divsf3 and __floatsisf.
// aie::to_float is a single fx2flt and aie::inv is native, so a reciprocal
// multiply through one vector lane keeps the row statistics off the libcalls.
static inline float scalar_mul(float a, float b) {
  return ::aie::mul(::aie::broadcast<float, 16>(a), b).to_vector<float>()[0];
}

// a - b * c, kept in the accumulator: subtracting two scalar_mul results
// forms a <16 x float> G_FSUB that Peano cannot legalize.
static inline float scalar_mul_sub(float a, float b, float c) {
  ::aie::accum<accfloat, 16> acc;
  acc.from_vector(::aie::broadcast<float, 16>(a));
  return ::aie::msc(acc, ::aie::broadcast<float, 16>(b), c)
      .to_vector<float>()[0];
}

// AIE2P has a scalar invsqrt. On AIE2 a scalar aie::invsqrt lowers to sqrtf,
// which does not link. The vector one is a bit-trick estimate good to 6.5e-4,
// so one Newton step, y * (1.5 - x / 2 * y * y), brings it to f32 accuracy.
static inline float scalar_invsqrt(float x) {
#if AIE_TUNED_AIE2P
  return ::aie::invsqrt(x);
#else
  ::aie::vector<float, 16> y = ::aie::invsqrt(::aie::broadcast<float, 16>(x));
  ::aie::vector<float, 16> half_xy = ::aie::mul(y, 0.5f * x).to_vector<float>();
  ::aie::accum<accfloat, 16> t;
  t.from_vector(::aie::broadcast<float, 16>(1.5f));
  t = ::aie::msc(t, half_xy, y);
  return ::aie::mul(y, t.to_vector<float>()).to_vector<float>()[0];
#endif
}

#endif
