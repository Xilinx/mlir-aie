//===- stack_size_recursive_kernel.cc ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_recursion_error.mlir and
// stack_size_explicit_override_wins.mlir. recurse calls itself, so its stack
// requirement is unbounded, and the measurement fails until the func.func
// declaration of the kernel carries a stack_size_override. Both functions are
// `extern "C"`, which fixes the symbol name that the recursion diagnostic
// prints; a `static` function keeps a C++-mangled name even inside `extern
// "C"`. The RUN lines of both tests compile at -O0, which keeps the
// self-recursion as a call and leaves the cycle in the call graph.

extern "C" void recurse(unsigned char *out, int n) {
  if (n > 0) {
    out[0] = static_cast<unsigned char>(n);
    recurse(out, n - 1);
  }
}

extern "C" void recursive_touch(unsigned char *out) { recurse(out, 8); }
