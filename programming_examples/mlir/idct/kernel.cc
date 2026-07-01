//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func1(int *a, int *b) {
  // int new_size = 64;
  // for (int i = 0; i < 64; i ++){
  //     b[i] = a[i] + 1;
  // }

  int val = a[3];
  int val2 = val + val;
  val2 += val;
  val2 += val;
  val2 += val;
  b[5] = val2;
}