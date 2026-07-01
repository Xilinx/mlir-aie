//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int *a, int *b, int size) {
  int new_size = 256;
  for (int i = 0; i < 256; i++) {
    b[i] = a[i];
  }
}