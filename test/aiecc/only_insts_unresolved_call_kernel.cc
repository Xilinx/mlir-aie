//===- only_insts_unresolved_call_kernel.cc ---------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `helper` is defined in no object the core links.

extern "C" {
void helper(int *p);

void kernel(int *p) {
  int local[64];
  for (int i = 0; i < 64; ++i)
    local[i] = p[i];
  helper(local);
}
}
