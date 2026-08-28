//===- rodata_overlay.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An overlay with a constant table, which must be refused.
//
// This is the realistic mistake: a kernel with a lookup table is perfectly
// ordinary, and nothing about it looks wrong. But .rodata is routed to *data*
// memory, and only .text travels on the overlay path -- so the table would
// never arrive, and the kernel would read whatever the last overlay left there.
// `volatile` on the index keeps the compiler from folding the lookup.

#include <cstdint>

static const int32_t kTable[64] = {
#define R8(n) n, n + 1, n + 2, n + 3, n + 4, n + 5, n + 6, n + 7,
    R8(0) R8(8) R8(16) R8(24) R8(32) R8(40) R8(48) R8(56)
#undef R8
};

extern "C" void overlay_entry(int32_t *in, int32_t *out) {
  volatile int32_t i = *in & 63;
  *out = kTable[i];
}
