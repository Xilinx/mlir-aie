//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

extern int32_t a_ping[256];
extern int32_t a_pong[256];
extern int32_t b_ping[256];
extern int32_t b_pong[256];

#define LOCK_OFFSET 48
#define A_WRITE (LOCK_OFFSET + 3)
#define A_READ (LOCK_OFFSET + 4)
#define B_WRITE (LOCK_OFFSET + 5)
#define B_READ (LOCK_OFFSET + 6)
#define DONE (LOCK_OFFSET + 7)

inline void func(int32_t *a, int32_t *b) {
  int val = a[3];
  int val2 = val + val;
  val2 += val;
  val2 += val;
  val2 += val;
  b[5] = val2;
}

extern "C" void core_7_3() {
  int bounds = 2; // iter;

  while (bounds > 0) {
    acquire_greater_equal(A_READ, 1);
    acquire_greater_equal(B_WRITE, 1);
    if ((bounds & 0x1) == 0) {
      func(a_ping, b_ping);
    } else {
      func(a_pong, b_pong);
    }
    release(A_WRITE, 1);
    release(B_READ, 1);
    bounds--;
  }
  acquire_equal(DONE, 1);
}
