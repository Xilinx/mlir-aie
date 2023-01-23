//===- kernel2.cc ------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func2(int *a, int *b) {
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
  b[6] = a[5] + a[5];
}