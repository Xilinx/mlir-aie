//===- large_stack_kernel.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Use more than one 16 KiB bank, leaving room in the 20 KiB reservation for
// call frames. Volatile indexed accesses keep the whole array on the stack.
extern "C" void large_stack(int *out) {
  volatile int scratch[4608];
  for (int i = 0; i < 4608; ++i)
    scratch[i] = i;
  out[0] = scratch[0] + scratch[4607];
}
