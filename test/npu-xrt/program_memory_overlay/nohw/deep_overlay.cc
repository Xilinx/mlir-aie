//===- deep_overlay.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An overlay with a stack frame far larger than the default 1024-byte budget.
//
// A 512-element local array is 2 KB. `volatile` keeps the compiler from
// promoting it to registers or eliding it -- without that the frame disappears
// and the test measures nothing.
//
// Nothing about this is unusual for a kernel: ffn_0 in yolo26n puts a 2 KB
// packing buffer on the stack for exactly this kind of reason. The hazard is
// that an overlay's frame is invisible to the resident's stack budget, which
// was fixed when the resident linked.

#include <cstdint>

extern "C" void overlay_entry(int32_t *in, int32_t *out) {
  volatile int32_t scratch[512];
  for (int i = 0; i < 512; i++)
    scratch[i] = in[i & 63] + i;
  int32_t acc = 0;
  for (int i = 0; i < 512; i++)
    acc += scratch[i];
  *out = acc;
}
