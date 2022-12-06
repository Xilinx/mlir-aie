//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
// #include <stdint.h>


//void func(int32_t *buf);

extern int32_t a[256];
extern int32_t b[256];

#define LOCK_OFFSET 48

extern "C" void core_1_3() {
  acquire(LOCK_OFFSET + 3, 1);
  acquire(LOCK_OFFSET + 5, 0);
  int val = a[3];
  int val2 = val + val;
  val2 += val;
  val2 += val;
  val2 += val;
  b[5] = val2;
  release(LOCK_OFFSET + 3, 0);
  release(LOCK_OFFSET + 5, 1);
}
