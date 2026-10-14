//===- resample_quantize.cc -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include "resample.h"

// Chunk core + k * CORES of the table, its weights at the precision the
// largest normalized weight `peak` sets. A size the table cannot hold writes
// [-1, window, 0, 0].
extern "C" void resample_quantize(int32_t *peak, int32_t *chunk, int32_t in,
                                  int32_t out, int32_t core, int32_t k,
                                  int32_t chunks) {
  event0();
  Axis a = axis(in, out, chunks * CORES);
  for (int i = 0; i < WORDS; i++)
    chunk[i] = 0;
  if (!a.ok) {
    chunk[0] = -1;
    chunk[1] = a.window;
    event1();
    return;
  }
  double wt_max = load(peak);
  int p;
  for (p = 0; p < 22; ++p)
    if ((int)(0.5 + wt_max * (1 << (p + 1))) >= (1 << 15))
      break;
  int first = (core + k * CORES) * a.per;
  int n = out - first;
  n = n < 0 ? 0 : n > a.per ? a.per : n;
  chunk[0] = p;
  chunk[1] = a.window;
  chunk[2] = first;
  chunk[3] = n;
  for (int s = 0; s < n; s++) {
    int32_t *slot = chunk + HEADER + s * a.slot;
    int xmin, xsize;
    double total = taps(a, first + s, xmin, xsize);
    slot[0] = xmin;
    slot[1] = xsize;
    int16_t *q = (int16_t *)(slot + 2);
    for (int j = 0; j < xsize; j++) {
      double v = (total != 0.0 ? w[j] / total : w[j]) * (1 << p);
      q[j] = v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v);
    }
  }
  event1();
}
