//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
// #include <stdint.h>

// void func(int32_t *buf);

extern int32_t a[256];
extern int32_t b[256];

#define LOCK_OFFSET 48

extern "C" void core_1_3() {
  acquire_greater_equal(LOCK_OFFSET + 3, 1);
  int val = a[3];
  int val2 = val + val;
  val2 += val;
  val2 += val;
  val2 += val;
  b[5] = val2;
  release(LOCK_OFFSET + 5, 1);
}
