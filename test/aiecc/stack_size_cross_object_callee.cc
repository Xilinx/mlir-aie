//===- stack_size_cross_object_callee.cc --------------------------*- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_cross_object.mlir. helper_cross's real,
// large-ish frame (~4096 bytes) has to be folded into entry_cross's path
// (stack_size_cross_object_caller.cc, a different object) for the analysis's
// cross-object attribution to be doing anything: if the edge were dropped,
// the computed requirement would silently fall back to entry_cross's own
// (trivial) frame instead.

extern "C" void helper_cross(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}
