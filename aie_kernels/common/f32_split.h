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

// AIE2 at 32 lanes trips Peano assertions (ISel, peephole) on these ops.
constexpr int f32_lanes = AIE_BF16_LANES;
using f32_acc = aie::accum<accfloat, f32_lanes>;
using bf16_vec = aie::vector<bfloat16, f32_lanes>;

static inline bf16_vec bf16_splat(int16_t bits) {
  return aie::broadcast<int16_t, f32_lanes>(bits).cast_to<bfloat16>();
}

// a = hi + mid + lo, exactly: each residual is taken in the accumulator, where
// a limb times one is exact. The core flushes bf16 subnormals, so this holds
// while lo, at least ulp(a), is normal: |a| from 2^-103.
static inline void split3(f32_acc a, bf16_vec &hi, bf16_vec &mid, bf16_vec &lo,
                          bf16_vec one) {
  hi = a.to_vector<bfloat16>();
  f32_acc r = aie::msc(a, hi, one);
  mid = r.to_vector<bfloat16>();
  lo = aie::msc(r, mid, one).to_vector<bfloat16>();
}

#endif
