//===- stack_size_same_named_static_a_kernel.cc -----------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_same_named_static_helpers.mlir. This `helper`
// has internal linkage, holds a real frame, and calls nothing.
// stack_size_same_named_static_b_kernel.cc defines an unrelated,
// self-recursive `helper` under the same name. One core links both objects, so
// the analysis keys the two apart.

static void helper(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

extern "C" void entry_a(unsigned char *out) { helper(out); }
