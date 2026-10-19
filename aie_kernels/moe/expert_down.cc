//===- expert_down.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One tile of a mixture-of-experts expert's down projection, weighted by the
// router and accumulated. Scalar, a first pass: neither vectorized nor
// checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef EXPERT_ROWS
#define EXPERT_ROWS 4
#endif
#ifndef EXPERT_WIDTH
#define EXPERT_WIDTH 256
#endif

// y[o] += weight[0] * (w row o . h), in float32. weight is the router's
// padded pair: weight[1] is never read.
extern "C" void expert_down(float *restrict h, bfloat16 *restrict w,
                            bfloat16 *restrict weight, float *restrict y) {
  event0();
  const float scale = (float)weight[0];
  for (int o = 0; o < EXPERT_ROWS; o++) {
    float acc = 0.0f;
    for (int k = 0; k < EXPERT_WIDTH; k++)
      acc += (float)w[o * EXPERT_WIDTH + k] * h[k];
    y[o] += scale * acc;
  }
  event1();
}
