//===- ctor_overlay.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An overlay carrying a static constructor, which must be refused.
//
// The initializer reads a volatile on purpose. A constructor the compiler can
// evaluate at compile time is folded away and emits no constructor section at
// all, so a test built on one would pass without ever exercising the check.

#include <cstdint>

static volatile int32_t seed = 3;

struct Table {
  int32_t v[4];
  Table() {
    for (int i = 0; i < 4; i++)
      v[i] = seed + i;
  }
};

static Table g_table;

extern "C" void overlay_entry(int32_t *in, int32_t *out) {
  *out = *in + g_table.v[2];
}
