//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int *a, int *b, int size) {
  for (int i = 0; i < size; i++) {
    int tmp = a[i];
    b[i] = tmp + 1;
  }
}
