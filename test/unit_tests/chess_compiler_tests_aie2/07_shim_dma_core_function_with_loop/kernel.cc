//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int *a, int *b, int size) {
  for (int i = 0; i < size; i++) {
    int tmp = a[i];
    b[i] = tmp + 1;
  }
}
