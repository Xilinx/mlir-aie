//===- resample.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// What resample_peak.cc and resample_quantize.cc agree on: one axis of
// torch's antialiased bicubic resize, IN samples to OUT, as a table of chunks
// of WORDS int32. A chunk is a HEADER-word header, [precision, window, first,
// outputs], then a slot per output, [start, count] and the window's int16
// weights two to a word. CORES cores take chunks core, core + CORES, ...; PER
// fixes a chunk's slots (0: as many as fit). The weights are float64, as
// torch computes them on the CPU, so every step is a soft-float IEEE double.

#ifndef AIE_KERNELS_VISION_RESAMPLE_H
#define AIE_KERNELS_VISION_RESAMPLE_H

#include <stdint.h>

#ifndef WORDS
#define WORDS 256
#endif
#ifndef CORES
#define CORES 16
#endif
#ifndef PER
#define PER 0
#endif

// torch's weights are float64 without contraction: an FMA changes the bits.
#pragma clang fp contract(off)

namespace {
constexpr int HEADER = 4;
constexpr int WMAX = 2 * (WORDS - HEADER - 2);

struct Axis {
  double scale, support, invscale;
  int in, out, window, slot, per;
  bool ok;
};

Axis axis(int32_t in, int32_t out, int32_t chunks) {
  Axis a = {};
  a.in = in;
  a.out = out;
  if (in < 1 || out < 1)
    return a;
  a.scale = (double)in / (double)out;
  a.support = a.scale >= 1.0 ? 2.0 * a.scale : 2.0;
  int s = (int)a.support;
  if ((double)s < a.support)
    s++;
  a.window = s * 2 + 1;
  a.invscale = a.scale >= 1.0 ? 1.0 / a.scale : 1.0;
  a.slot = 2 + (a.window + 1) / 2;
  a.per = (WORDS - HEADER) / a.slot;
  if (PER)
    a.per = PER <= a.per ? PER : 0;
  a.ok = a.per >= 1 && a.per * chunks >= out;
  return a;
}

double filter(double x) {
  const double A = -0.5;
  x = __builtin_fabs(x);
  if (x < 1.0)
    return ((A + 2) * x - (A + 3)) * x * x + 1;
  if (x < 2.0)
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  return 0.0;
}

double w[WMAX];

// Output i's unnormalized weights in w; returns their sequential sum.
double taps(const Axis &a, int i, int &xmin, int &xsize) {
  double center = a.scale * (i + 0.5);
  int lo = (int)(center - a.support + 0.5);
  xmin = lo > 0 ? lo : 0;
  int hi = (int)(center + a.support + 0.5);
  if (hi > a.in)
    hi = a.in;
  xsize = hi - xmin;
  if (xsize < 0)
    xsize = 0;
  if (xsize > a.window)
    xsize = a.window;
  double total = 0.0;
  for (int j = 0; j < xsize; j++) {
    w[j] = filter(((double)(j + xmin) - center + 0.5) * a.invscale);
    total += w[j];
  }
  return total;
}

double load(const int32_t *p) {
  double v;
  __builtin_memcpy(&v, p, sizeof v);
  return v;
}
} // namespace

#endif // AIE_KERNELS_VISION_RESAMPLE_H
