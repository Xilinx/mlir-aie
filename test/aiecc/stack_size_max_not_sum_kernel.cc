//===- stack_size_max_not_sum_kernel.cc -------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_max_not_sum.mlir. entry_a and entry_b each hold
// a local array of about 4096 bytes, which sets a frame far above any overhead
// that varies with the compiler version. The core body calls the two in
// sequence, and neither calls the other, so one frame is live at a time and
// the requirement of the core is about max(frame_a, frame_b), about 4096
// bytes. The volatile writes keep both arrays.

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
