//===- stack_size_max_not_sum_kernel.cc -------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_max_not_sum.mlir. entry_a and entry_b each
// have their own ~4096-byte local array, forcing a real, large frame that
// dwarfs any compiler-version-dependent overhead. The core body calls both,
// sequentially and independently (neither calls the other) -- since only
// one of the two can be on the stack at a time, the core's requirement must
// be roughly max(frame_a, frame_b) (~4096 bytes), not their sum (~8192
// bytes). volatile writes keep the compiler from optimizing the arrays away
// entirely.

extern "C" void entry_a(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

extern "C" void entry_b(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[1] = out[0];
  out[0] = buf[1];
}
