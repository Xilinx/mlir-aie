//===- kernel3.cc ------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func3(int *a, int *b) {
  // int new_size = 64;
  // for (int i = 0; i < 64; i ++){
  //     b[i] = a[i] + 1;
  // }

  // int val=a[3];
  // int val2=val+val;
  // val2 += val;
  // val2 += val;
  // val2 += val;
  // b[5] = val2;
  b[5] = a[5];
  b[6] = a[6];
  b[7] = a[6] * 3;
}