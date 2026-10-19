// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_SCALAR_EXP_H
#define AIE_KERNELS_COMMON_SCALAR_EXP_H

#include <stdint.h>
#include <string.h>

// exp(x) in plain scalar float32, for kernels that call it a few times per
// row; small enough to leave room in program memory, unlike exp64.h's
// doubles. x = k ln 2 + r with |r| <= ln 2 / 2 (Cody-Waite: ln 2's high part
// has 12 bits, so k * high is exact), e^r by its Taylor series to r^7
// (6e-9 at the ends), times 2^k. Below -87.3 it returns 0, where the result
// would be subnormal; above 88.7, infinity.
static inline float scalar_exp(float x) {
  if (x < -87.3f)
    return 0.0f;
  if (x > 88.7f)
    return __builtin_inff();
  const float t = x * 1.44269504f;
  const int k = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
  const float r = (x - k * 0.693145751953125f) - k * 1.42860677e-6f;
  float p = 1.0f / 5040;
  p = p * r + 1.0f / 720;
  p = p * r + 1.0f / 120;
  p = p * r + 1.0f / 24;
  p = p * r + 1.0f / 6;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  // 2^k as its bits; k reaches 128 only just below 88.7, so halve it there.
  const int half = k > 127;
  const uint32_t bits = (uint32_t)(k - half + 127) << 23;
  float scale;
  memcpy(&scale, &bits, sizeof(scale));
  return half ? p * scale * 2.0f : p * scale;
}

#endif
