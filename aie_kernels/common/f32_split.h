//===- f32_split.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_F32_SPLIT_H
#define AIE_KERNELS_COMMON_F32_SPLIT_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// f32 arithmetic from bf16 products, on any architecture: three bf16 limbs
// hold an f32 exactly, and the product of two limbs is exact in the f32
// accumulator, so an f32 product is a sum of limb products.

// The limb chains are latency-bound, so AIE2 too runs them fastest at 32 lanes.
constexpr int f32_lanes = 32;
using f32_acc = aie::accum<accfloat, f32_lanes>;
using bf16_vec = aie::vector<bfloat16, f32_lanes>;

template <unsigned N = f32_lanes>
static inline aie::vector<bfloat16, N> bf16_splat(int16_t bits) {
  return aie::broadcast<int16_t, N>(bits).template cast_to<bfloat16>();
}

// a = hi + mid + lo, exactly: each residual is taken in the accumulator, where
// a limb times one is exact. The core flushes bf16 subnormals, so this holds
// while lo, at least ulp(a), is normal: |a| from 2^-103.
template <unsigned N>
static inline void
split3(aie::accum<accfloat, N> a, aie::vector<bfloat16, N> &hi,
       aie::vector<bfloat16, N> &mid, aie::vector<bfloat16, N> &lo,
       aie::vector<bfloat16, N> one) {
  hi = a.template to_vector<bfloat16>();
#if AIE_ARCH_AIE2
  // AIE2's accumulator rounds a to hi's exponent when hi rounds up past a
  // power of two; taking hi away in halves stays within a's.
  const aie::vector<bfloat16, N> half = bf16_splat<N>(0x3f00);
  aie::accum<accfloat, N> r = aie::msc(aie::msc(a, hi, half), hi, half);
#else
  aie::accum<accfloat, N> r = aie::msc(a, hi, one);
#endif
  mid = r.template to_vector<bfloat16>();
  lo = aie::msc(r, mid, one).template to_vector<bfloat16>();
}

#endif
