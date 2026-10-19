//===- selective_scan.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Mamba2's state space recurrence for one head over a block of rows. Scalar,
// a first pass: neither vectorized nor checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../sample/exp64.h"
#include "softplus.h"

#ifndef SCAN_ROWS
#define SCAN_ROWS 16
#endif
#ifndef SCAN_HEAD_DIM
#define SCAN_HEAD_DIM 16
#endif
#ifndef SCAN_BLOCKS
#define SCAN_BLOCKS 1
#endif
#ifndef SCAN_STATE
#define SCAN_STATE 32
#endif

namespace {

float s[SCAN_HEAD_DIM * SCAN_STATE];

} // namespace

// block is x (SCAN_ROWS x SCAN_HEAD_DIM), then bc (SCAN_ROWS rows of the
// group's B then C, SCAN_STATE each), then dt (SCAN_ROWS). head is
// (dt_bias, A, D), then the state before the head's first block. b counts
// blocks, SCAN_BLOCKS to a head, and block b loads the state when it is a
// head's first. Each block runs its first valid rows: step = softplus(dt +
// dt_bias), state = exp(step * A) * state + step * outer(x, B), y = state @ C +
// D * x. The rows after valid leave the state alone and write zeros. The state
// stays in the core from block to block; final is it after this block.
extern "C" void selective_scan(bfloat16 *restrict block, float *restrict head,
                               float *restrict final, bfloat16 *restrict y,
                               int32_t valid, int32_t b) {
  event0();
  const bfloat16 *x = block;
  const bfloat16 *bc = x + SCAN_ROWS * SCAN_HEAD_DIM;
  const bfloat16 *dt = bc + SCAN_ROWS * 2 * SCAN_STATE;
  if (b % SCAN_BLOCKS == 0)
    for (int i = 0; i < SCAN_HEAD_DIM * SCAN_STATE; i++)
      s[i] = head[3 + i];
  const float dt_bias = head[0], A = head[1], D = head[2];
  for (int t = 0; t < SCAN_ROWS; t++) {
    bfloat16 *yt = y + t * SCAN_HEAD_DIM;
    if (t >= valid) {
      for (int p = 0; p < SCAN_HEAD_DIM; p++)
        yt[p] = 0.0f;
      continue;
    }
    const bfloat16 *xt = x + t * SCAN_HEAD_DIM;
    const bfloat16 *B = bc + t * 2 * SCAN_STATE;
    const bfloat16 *C = B + SCAN_STATE;
    const float step = softplus((float)dt[t] + dt_bias);
    const float decay = (float)exp64((double)(step * A));
    for (int p = 0; p < SCAN_HEAD_DIM; p++) {
      const float xp = (float)xt[p];
      const float sx = step * xp;
      float *sp = s + p * SCAN_STATE;
      float acc = 0.0f;
      for (int n = 0; n < SCAN_STATE; n++) {
        sp[n] = decay * sp[n] + sx * (float)B[n];
        acc += sp[n] * (float)C[n];
      }
      yt[p] = acc + D * xp;
    }
  }
  for (int i = 0; i < SCAN_HEAD_DIM * SCAN_STATE; i++)
    final[i] = s[i];
  event1();
}
