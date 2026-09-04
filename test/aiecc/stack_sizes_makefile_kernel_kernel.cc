//===- stack_sizes_makefile_kernel_kernel.cc --------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_sizes_makefile_kernel.test.

extern "C" void add_one(int *a, int *b, int n) {
  for (int i = 0; i < n; i++)
    b[i] = a[i] + 1;
}
