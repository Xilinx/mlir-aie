//===- stack_size_cross_object_callee.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_cross_object.mlir. The frame of helper_cross
// holds about 4096 bytes. entry_cross, in stack_size_cross_object_caller.cc,
// calls helper_cross from another object, so the computed requirement covers
// that frame only when the analysis attributes the call across objects.

extern "C" void helper_cross(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}
