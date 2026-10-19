//===- gated_delta_rule.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Gated DeltaNet's recurrence for one value head over a block of rows.
// Scalar, a first pass: neither vectorized nor checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "../common/scalar_exp.h"
#include "softplus.h"

#ifndef GDR_ROWS
#define GDR_ROWS 16
#endif
#ifndef GDR_KEY_DIM
#define GDR_KEY_DIM 32
#endif
#ifndef GDR_VALUE_DIM
#define GDR_VALUE_DIM 16
#endif
#ifndef GDR_BLOCKS
#define GDR_BLOCKS 1
#endif

namespace {
constexpr int R = GDR_ROWS, Dk = GDR_KEY_DIM, Dv = GDR_VALUE_DIM;

float s[Dk * Dv];
float delta[Dv];

// 1 / sqrt(x) for x > 0: the bit-trick estimate and three Newton steps.
float rsqrt_newton(float x) {
  uint32_t i;
  memcpy(&i, &x, sizeof(i));
  i = 0x5f3759df - (i >> 1);
  float y;
  memcpy(&y, &i, sizeof(y));
  for (int n = 0; n < 3; n++)
    y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

// scale / |m|, the eps inside the root as the reference has it.
float inv_norm(const bfloat16 *m, float scale) {
  float sq = 0.0f;
  for (int i = 0; i < Dk; i++)
    sq += (float)m[i] * (float)m[i];
  return rsqrt_newton(sq + 1e-6f) * scale;
}
} // namespace

// block is q (R x Dk), k (R x Dk), v (R x Dv), then a (R) and b (R); head is
// (dt_bias, A), then the state (Dk x Dv) before the head's first block. b_idx
// counts blocks, GDR_BLOCKS to a head, and a head's first block loads the
// state. Each block runs its first valid rows: q and k are l2-normalized
// (eps 1e-6) and q scaled by 1 / sqrt(Dk), g = A * softplus(a + dt_bias),
// beta = sigmoid(b), state = exp(g) * state, then state += outer(k, beta *
// (v - k @ state)), and y = q @ state. The rows after valid leave the state
// alone and write zeros; final is the state after this block.
extern "C" void gated_delta_rule(bfloat16 *restrict block, float *restrict head,
                                 float *restrict final, bfloat16 *restrict y,
                                 int32_t valid, int32_t b_idx) {
  event0();
  const bfloat16 *q = block, *k = q + R * Dk, *v = k + R * Dk;
  const bfloat16 *a = v + R * Dv, *b = a + R;
  if (b_idx % GDR_BLOCKS == 0)
    for (int i = 0; i < Dk * Dv; i++)
      s[i] = head[2 + i];
  const float dt_bias = head[0], A = head[1];
  const float q_scale = rsqrt_newton((float)Dk);
  for (int t = 0; t < R; t++) {
    bfloat16 *yt = y + t * Dv;
    if (t >= valid) {
      for (int j = 0; j < Dv; j++)
        yt[j] = 0.0f;
      continue;
    }
    const bfloat16 *qt = q + t * Dk, *kt = k + t * Dk, *vt = v + t * Dv;
    const float qn = inv_norm(qt, q_scale), kn = inv_norm(kt, 1.0f);
    const float decay =
        scalar_exp(A * softplus((float)a[t] + dt_bias));
    const float beta = 1.0f / (1.0f + scalar_exp(-(float)b[t]));
    for (int i = 0; i < Dk * Dv; i++)
      s[i] *= decay;
    for (int j = 0; j < Dv; j++) {
      float kv = 0.0f;
      for (int i = 0; i < Dk; i++)
        kv += (float)kt[i] * kn * s[i * Dv + j];
      delta[j] = beta * ((float)vt[j] - kv);
    }
    for (int i = 0; i < Dk; i++) {
      const float ki = (float)kt[i] * kn;
      for (int j = 0; j < Dv; j++)
        s[i * Dv + j] += ki * delta[j];
    }
    for (int j = 0; j < Dv; j++) {
      float acc = 0.0f;
      for (int i = 0; i < Dk; i++)
        acc += (float)qt[i] * qn * s[i * Dv + j];
      yt[j] = acc;
    }
  }
  for (int i = 0; i < Dk * Dv; i++)
    final[i] = s[i];
  event1();
}
