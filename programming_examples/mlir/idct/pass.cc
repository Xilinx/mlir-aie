//===- pass.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void pass(int *a, int *b) {
  for (int i = 0; i < 64; i++) {
    b[i] = a[i];
  }
}