//===- stack_size_same_named_static_a_kernel.cc ------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_same_named_static_helpers.mlir. `helper` here
// is a plain (non-recursive) internal-linkage function with a real frame; a
// sibling object (stack_size_same_named_static_b_kernel.cc) defines an
// unrelated, self-recursive `helper` under the identical name. Both objects
// are linked into the same core so the analysis must not let the two alias.

static void helper(unsigned char *out) {
  volatile unsigned char buf[4096];
  buf[0] = out[0];
  out[0] = buf[0];
}

extern "C" void entry_a(unsigned char *out) { helper(out); }
