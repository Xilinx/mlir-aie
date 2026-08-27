//===- stack_size_same_named_static_b_kernel.cc ------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_same_named_static_helpers.mlir. `helper` here
// is unrelated to (and self-recursive, unlike) the `helper` defined in
// stack_size_same_named_static_a_kernel.cc -- the two happen to share both a
// name and a signature (so they mangle identically: a `static` C++ function
// is mangled from its name and parameter types regardless of `extern "C"`,
// see stack_size_recursive_kernel.cc) only because they are internal-linkage
// symbols in separate objects.

static void helper(unsigned char *out) {
  helper(out);
  out[0] = 1;
}

extern "C" void unused_entry_b(unsigned char *out) { helper(out); }
