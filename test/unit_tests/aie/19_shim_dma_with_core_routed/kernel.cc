//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2021 Xilinx, Inc.
// Copyright (C) 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int *a, int *b, int size) {
  int new_size = 256;
  for (int i = 0; i < 256; i++) {
    b[i] = a[i];
  }
}
