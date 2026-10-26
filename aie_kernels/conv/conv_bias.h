//===- conv_bias.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the int8 convolutions with a bias share: their rounding, and the
// accumulator each output block starts from.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_CONV_CONV_BIAS_H
#define AIE_KERNELS_CONV_CONV_BIAS_H

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../aie_kernel_utils.h"

#ifdef CONV_ROUND_EVEN
#define CONV_BIAS_ROUNDING aie::rounding_mode::conv_even
#else
#define CONV_BIAS_ROUNDING aie::rounding_mode::symmetric_inf
#endif

// M pixels' accumulators, each starting from the same 8 channels' biases.
template <int M>
static inline __attribute__((always_inline)) aie::accum<acc32, M * 8>
bias_seed(aie::vector<int32, 8> bias) {
  aie::vector<int32, M * 8> seed;
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < M; p++)
    seed.insert(p, bias);
  aie::accum<acc32, M * 8> seeded;
  seeded.from_vector(seed, 0);
  return seeded;
}

#endif
