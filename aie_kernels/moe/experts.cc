//===- experts.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One row of a mixture of experts' feed-forward on one core, from two input
// streams: the row, and one stream of weight tiles. Scalar, a first pass:
// neither vectorized nor checked on a device yet.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../sample/exp64.h"

#ifndef EXPERTS_HIDDEN
#define EXPERTS_HIDDEN 64
#endif
#ifndef EXPERTS_WIDTH
#define EXPERTS_WIDTH 32
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

float acc[2 * I]; // gate and up dot products, then h in acc[0, I)
float sum[H];     // the weighted sum over the chosen experts
} // namespace

// row is x (H), then the router's 2 * TOP_K weights, chosen expert j's at
// 2 * j. Step s of a row brings tile s of its stream: for each chosen expert
// in turn, the T x C blocks of gate_up (2I x H) row block by row block, then
// those of down (H x I). y is the row's bf16 output so far; after the last
// step, sum over j of weight j times down @ (silu(g) * u).
extern "C" void experts(bfloat16 *restrict row, bfloat16 *restrict tile,
                        bfloat16 *restrict y, int32_t step) {
  event0();
  const int j = step / expert_steps;
  int s = step - j * expert_steps;
  if (step == 0)
    for (int o = 0; o < H; o++)
      sum[o] = 0.0f;
  if (s == 0)
    for (int r = 0; r < 2 * I; r++)
      acc[r] = 0.0f;
  if (s < gate_up_steps) {
    const int r0 = s / (H / C) * T, c0 = s % (H / C) * C;
    for (int r = 0; r < T; r++)
      for (int k = 0; k < C; k++)
        acc[r0 + r] += (float)tile[r * C + k] * (float)row[c0 + k];
    if (s == gate_up_steps - 1)
      for (int i = 0; i < I; i++) {
        const float g = acc[i];
        acc[i] = g / (1.0f + (float)exp64(-(double)g)) * acc[I + i];
      }
  } else {
    s -= gate_up_steps;
    const int r0 = s / (I / C) * T, c0 = s % (I / C) * C;
    const float weight = (float)row[H + 2 * j];
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
