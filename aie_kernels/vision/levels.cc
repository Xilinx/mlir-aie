//===- levels.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A row of uint8 RGB pixels as bf16 pixels of four channels, through
// per-channel tables a host fills exactly (an image processor's rescale and
// normalization), a run of them placed in a row of fill.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef BGR
#define BGR 0
#endif

namespace {
constexpr int LEVELS = 256;
constexpr int CHANNELS = 4;
} // namespace

extern "C" {

// table: three channels' LEVELS entries, then the fill's CHANNELS values.
void levelsLine(const uint8_t *__restrict in, const bfloat16 *__restrict table,
                bfloat16 *__restrict out, int32_t first, int32_t count,
                int32_t at, int32_t row, int32_t lo, int32_t hi) {
  event0();
  const bfloat16 *fill = table + 3 * LEVELS;
  const bfloat16 f0 = fill[0], f1 = fill[1], f2 = fill[2], f3 = fill[3];
  for (int x = 0; x < OUT_WIDTH; x++) {
    out[CHANNELS * x] = f0;
    out[CHANNELS * x + 1] = f1;
    out[CHANNELS * x + 2] = f2;
    out[CHANNELS * x + 3] = f3;
  }
  // Nothing past either row: a count, first or at outside them takes less.
  if (first < 0 || at < 0 || row < lo || row >= hi)
    count = 0;
  if (count > OUT_WIDTH - at)
    count = OUT_WIDTH - at;
  if (count > IN_BYTES / 3 - first)
    count = IN_BYTES / 3 - first;
  const uint8_t *p = in + 3 * first;
  bfloat16 *q = out + CHANNELS * at;
  for (int i = 0; i < count; i++) {
    q[0] = table[p[BGR ? 2 : 0]];
    q[1] = table[LEVELS + p[1]];
    q[2] = table[2 * LEVELS + p[BGR ? 0 : 2]];
    p += 3;
    q += CHANNELS;
  }
  event1();
}

} // extern "C"
