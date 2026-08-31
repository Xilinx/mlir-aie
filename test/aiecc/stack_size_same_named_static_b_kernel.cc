//===- stack_size_same_named_static_b_kernel.cc -----------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_same_named_static_helpers.mlir. This `helper`
// calls itself, and is unrelated to the `helper` in
// stack_size_same_named_static_a_kernel.cc. The two share a name and a
// signature, so they mangle to one symbol name: C++ mangles a `static`
// function from its name and its parameter types, see
// stack_size_recursive_kernel.cc. Internal linkage keeps each one inside its
// own object.

static void helper(unsigned char *out) {
  helper(out);
  out[0] = 1;
}

extern "C" void unused_entry_b(unsigned char *out) { helper(out); }
