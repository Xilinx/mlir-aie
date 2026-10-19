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

// Comparisons, conversions and power-of-two scalings below work on the bit
// pattern: exact for the finite doubles these kernels make, and no libcall.
constexpr uint64_t SIGN = 1ull << 63;
constexpr uint64_t FRAC = (1ull << 52) - 1;

uint64_t bits(double x) { return __builtin_bit_cast(uint64_t, x); }
double from_bits(uint64_t u) { return __builtin_bit_cast(double, u); }
bool is_zero(double x) { return (bits(x) << 1) == 0; }

// Orders finite doubles as < does, with -0 equal to +0.
int64_t key(double x) {
  uint64_t u = bits(x);
  int64_t m = (int64_t)(u & ~SIGN);
  return u & SIGN ? -m : m;
}

double to_double(int32_t n) {
  if (n == 0)
    return 0.0;
  uint32_t u = n < 0 ? 0u - (uint32_t)n : (uint32_t)n;
  int e = 31 - __builtin_clz(u);
  uint64_t sign = n < 0 ? SIGN : 0;
  return from_bits(sign | (uint64_t)(e + 1023) << 52 |
                   (((uint64_t)u << (52 - e)) & FRAC));
}

// (int)x, truncating toward zero.
int32_t to_int(double x) {
  uint64_t u = bits(x);
  int e = (int)((u >> 52) & 0x7ff) - 1023;
  if (e < 0)
    return 0;
  if (e > 30)
    return (int32_t)x;
  int32_t r = (int32_t)(((u & FRAC) | (1ull << 52)) >> (52 - e));
  return u & SIGN ? -r : r;
}

// x * 2^p for 0 <= p < 31: a normal x moves only its exponent.
double scale2(double x, int p) {
  uint64_t u = bits(x);
  uint64_t ef = (u >> 52) & 0x7ff;
  if ((u << 1) == 0)
    return x;
  if (ef == 0 || ef + p >= 0x7ff)
    return x * (double)(1 << p);
  return from_bits(u + ((uint64_t)p << 52));
}

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
  uint64_t u = bits(x) & ~SIGN;
  x = from_bits(u);
  if (u < bits(1.0))
    return ((A + 2) * x - (A + 3)) * x * x + 1;
  if (u < bits(2.0))
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  return 0.0;
}

double w[WMAX];

// Output i's unnormalized weights in w; returns their sequential sum.
double taps(const Axis &a, int i, int &xmin, int &xsize) {
  double center = a.scale * (to_double(i) + 0.5);
  int lo = to_int(center - a.support + 0.5);
  xmin = lo > 0 ? lo : 0;
  int hi = to_int(center + a.support + 0.5);
  if (hi > a.in)
    hi = a.in;
  xsize = hi - xmin;
  if (xsize < 0)
    xsize = 0;
  if (xsize > a.window)
    xsize = a.window;
  double total = 0.0;
  for (int j = 0; j < xsize; j++) {
    w[j] = filter((to_double(j + xmin) - center + 0.5) * a.invscale);
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
