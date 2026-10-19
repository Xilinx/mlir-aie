//===- softplus.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Scalar float32 softplus for the state space kernels' step sizes.

#ifndef AIE_KERNELS_SSM_SOFTPLUS_H
#define AIE_KERNELS_SSM_SOFTPLUS_H

#include "../common/scalar_exp.h"

// log_f32.cc's log1p(f) / f for f in [sqrt(1/2) - 1, sqrt(2) - 1).
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

// log(1 + exp(z)) as max(z, 0) + log1p(u), u = exp(-|z|) in (0, 1]. For
// 1 + u >= sqrt(2), log1p(u) is log 2 plus log of (1 + u) / 2 = 1 + f.
static inline float softplus(float z) {
  const float a = z < 0.0f ? -z : z;
  const float u = scalar_exp(-a);
  const bool halve = u >= 0.41421356f;
  const float f = halve ? 0.5f * (u - 1.0f) : u;
  float q = log_q[9];
  for (int k = 8; k >= 0; k--)
    q = q * f + log_q[k];
  return (z > 0.0f ? z : 0.0f) + f * q + (halve ? 0.69314718f : 0.0f);
}

#endif
