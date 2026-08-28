//===- kernel.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A scalar byte-store loop, which is all this takes.
//
// The trip count must stay a compile-time constant and the loop must stay
// inlined -- that is what makes the compiler unroll it, and the unrolled form is
// what goes wrong. Passing the count as a runtime parameter produces correct
// code and a test that proves nothing.

#include <cstdint>

extern "C" void fill_tile(int8_t *t) {
  for (int i = 0; i < 1024; i++)
    t[i] = 0x11;
}
