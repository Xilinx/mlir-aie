//===- experts.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A mixture of experts' feed-forward on one core, a row at a time, from two
// input streams: the row, and every expert's weight tiles in a fixed order.
// Scalar, a first pass: neither vectorized nor checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "../common/scalar_exp.h"

#ifndef EXPERTS_HIDDEN
#define EXPERTS_HIDDEN 64
#endif
#ifndef EXPERTS_WIDTH
#define EXPERTS_WIDTH 32
#endif
#ifndef EXPERTS_EXPERTS
#define EXPERTS_EXPERTS 4
#endif
#ifndef EXPERTS_TOP_K
#define EXPERTS_TOP_K 2
#endif
#ifndef EXPERTS_TILE_ROWS
#define EXPERTS_TILE_ROWS 8
#endif
#ifndef EXPERTS_TILE_COLS
#define EXPERTS_TILE_COLS 32
#endif

namespace {
constexpr int H = EXPERTS_HIDDEN, I = EXPERTS_WIDTH;
constexpr int T = EXPERTS_TILE_ROWS, C = EXPERTS_TILE_COLS;
static_assert((2 * I) % T == 0 && H % T == 0, "tile rows must divide 2I and H");
static_assert(H % C == 0 && I % C == 0, "tile columns must divide H and I");

constexpr int gate_up_steps = (2 * I / T) * (H / C);
constexpr int expert_steps = gate_up_steps + (H / T) * (I / C);
constexpr int row_steps = EXPERTS_EXPERTS * expert_steps;

float acc[2 * I]; // gate and up dot products, then h in acc[0, I)
float sum[H];     // the weighted sum over the chosen experts
} // namespace

// row is x (H), then the router's 2 * TOP_K weights (chosen expert j's at
// 2 * j), then its TOP_K int32 ids. Step s brings tile s % row_steps of the
// row's stream: for each expert e in order, the T x C blocks of gate_up
// (2I x H) row block by row block, then those of down (H x I). The tiles of
// an expert the row did not choose are skipped. y is the row's bf16 output
// so far; after the row's last step, the sum over j of weight j times
// down[ids[j]] @ (silu(g) * u).
extern "C" void experts(bfloat16 *restrict row, bfloat16 *restrict tile,
                        bfloat16 *restrict y, int32_t step) {
  event0();
  step %= row_steps;
  const int e = step / expert_steps;
  int s = step - e * expert_steps;
  if (step == 0)
    for (int o = 0; o < H; o++)
      sum[o] = 0.0f;
  int32_t ids[EXPERTS_TOP_K];
  memcpy(ids, row + H + 2 * EXPERTS_TOP_K, sizeof(ids));
  float weight = 0.0f;
  bool chosen = false;
  for (int j = 0; j < EXPERTS_TOP_K; j++)
    if (ids[j] == e) {
      weight += (float)row[H + 2 * j];
      chosen = true;
    }
  if (chosen && s == 0)
    for (int r = 0; r < 2 * I; r++)
      acc[r] = 0.0f;
  if (chosen && s < gate_up_steps) {
    const int r0 = s / (H / C) * T, c0 = s % (H / C) * C;
    for (int r = 0; r < T; r++)
      for (int k = 0; k < C; k++)
        acc[r0 + r] += (float)tile[r * C + k] * (float)row[c0 + k];
    if (s == gate_up_steps - 1)
      for (int i = 0; i < I; i++) {
        const float g = acc[i];
        acc[i] = g / (1.0f + scalar_exp(-g)) * acc[I + i];
      }
  } else if (chosen) {
    s -= gate_up_steps;
    const int r0 = s / (I / C) * T, c0 = s % (I / C) * C;
    for (int r = 0; r < T; r++) {
      float dot = 0.0f;
      for (int k = 0; k < C; k++)
        dot += (float)tile[r * C + k] * acc[c0 + k];
      sum[r0 + r] += weight * dot;
    }
  }
  for (int o = 0; o < H; o++)
    y[o] = sum[o];
  event1();
}
