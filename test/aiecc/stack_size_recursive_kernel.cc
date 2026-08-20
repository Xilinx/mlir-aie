//===- stack_size_recursive_kernel.cc ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_recursion_error.mlir and
// stack_size_explicit_override_wins.mlir. recurse calls itself, so its stack
// requirement is fundamentally unbounded -- exactly the case that must fail
// auto-measurement unless the kernel's func.func declaration carries
// stack_size_override. Both functions are `extern "C"` (a `static` function
// still gets a C++-mangled name even inside `extern "C"`, which would make
// the recursion diagnostic's symbol name unpredictable) and compiled at -O0
// (set by both tests' RUN lines) so the compiler can't turn the
// self-recursion into a loop, which would hide the cycle entirely.

extern "C" void recurse(unsigned char *out, int n) {
  if (n > 0) {
    out[0] = static_cast<unsigned char>(n);
    recurse(out, n - 1);
  }
}

extern "C" void recursive_touch(unsigned char *out) { recurse(out, 8); }
