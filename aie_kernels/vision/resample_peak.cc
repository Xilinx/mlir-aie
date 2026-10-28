//===- resample_peak.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include "resample.h"

// The largest normalized weight of the outputs in chunks core, core + CORES,
// ...: division rounds monotonically, so it is the largest weight over the
// total (the smallest, for a negative total), one division per output.
extern "C" void resample_peak(int32_t *peak, int32_t in, int32_t out,
                              int32_t core, int32_t chunks) {
  event0();
  Axis a = axis(in, out, chunks * CORES);
  double m = 0.0;
  int64_t km = 0;
  for (int k = 0; a.ok && k < chunks; k++) {
    int first = (core + k * CORES) * a.per;
    int last = first + a.per < out ? first + a.per : out;
    for (int i = first; i < last; i++) {
      int xmin, xsize;
      double total = taps(a, i, xmin, xsize);
      if (is_zero(total))
        continue;
      // A negative total wants the smallest weight: compare negated keys.
      bool negative = key(total) < 0;
      double e = w[0];
      int64_t ke = negative ? -key(e) : key(e);
      for (int j = 1; j < xsize; j++) {
        int64_t kj = negative ? -key(w[j]) : key(w[j]);
        if (ke < kj) {
          e = w[j];
          ke = kj;
        }
      }
      double v = e / total;
      int64_t kv = key(v);
      if (km < kv) {
        m = v;
        km = kv;
      }
    }
  }
  __builtin_memcpy(peak, &m, sizeof m);
  event1();
}

// The larger of two peaks, for a chain of cores to reduce them.
extern "C" void resample_join(int32_t *a, int32_t *b, int32_t *out) {
  double x = load(a), y = load(b);
  double m = key(x) < key(y) ? y : x;
  __builtin_memcpy(out, &m, sizeof m);
}
