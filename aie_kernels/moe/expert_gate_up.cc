//===- expert_gate_up.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One tile of a mixture-of-experts expert's gated activation. Scalar, a first
// pass: neither vectorized nor checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../sample/exp64.h"

#ifndef EXPERT_ROWS
#define EXPERT_ROWS 4
#endif
#ifndef EXPERT_HIDDEN
#define EXPERT_HIDDEN 256
#endif

// w holds EXPERT_ROWS gate rows, then the EXPERT_ROWS up rows they pair with.
// h[i] = silu(g) * u for g and u gate and up row i's dot products with x, in
// float32.
extern "C" void expert_gate_up(bfloat16 *restrict x, bfloat16 *restrict w,
                               float *restrict h) {
  event0();
  for (int i = 0; i < EXPERT_ROWS; i++) {
    const bfloat16 *gate = w + i * EXPERT_HIDDEN;
    const bfloat16 *up = gate + EXPERT_ROWS * EXPERT_HIDDEN;
    float g = 0.0f, u = 0.0f;
    for (int k = 0; k < EXPERT_HIDDEN; k++) {
      float xk = (float)x[k];
      g += (float)gate[k] * xk;
      u += (float)up[k] * xk;
    }
    h[i] = g / (1.0f + (float)exp64(-(double)g)) * u;
  }
  event1();
}
