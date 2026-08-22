//===- kernels.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The overlay kernels. One source, compiled once per OVL_ID, each producing a
// different computation behind the same entry symbol -- the resident calls a
// single fixed address, and which kernel is there is decided at run time by
// writing program memory.
//
// Each is linked on its own at the slot address (see overlay.py), so unlike the
// pairs in ../pm_write_while_running these are ordinary code: they may branch,
// call resident helpers, and be any size that fits the slot.

#include <cstdint>

#ifndef OVL_ID
#error "OVL_ID must be defined; each overlay is a separate compilation"
#endif

// Supplied by the resident image and resolved through --just-symbols, so an
// overlay can call back into code that is always present rather than carrying
// its own copy.
extern "C" int32_t ovl_bias(void);

extern "C" void overlay_entry(int32_t *in, int32_t *out, int32_t n) {
  for (int32_t i = 0; i < n; i++) {
#if OVL_ID == 0
    out[i] = in[i] + ovl_bias();
#elif OVL_ID == 1
    out[i] = in[i] * 3;
#elif OVL_ID == 2
    out[i] = -in[i];
#else
#error "unknown OVL_ID"
#endif
  }
}
