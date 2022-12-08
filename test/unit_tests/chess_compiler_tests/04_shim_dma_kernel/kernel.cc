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

//int32_t iter;

extern int32_t a_ping[256];
extern int32_t a_pong[256];
extern int32_t b_ping[256];
extern int32_t b_pong[256];

#define LOCK_OFFSET 48
#define A_PING (LOCK_OFFSET+3)
#define A_PONG (LOCK_OFFSET+4)
#define B_PING (LOCK_OFFSET+5)
#define B_PONG (LOCK_OFFSET+6)
#define LOCK_READ  1
#define LOCK_WRITE 0


inline void func(int32_t *a, int32_t *b)
{
    int val=a[3];
    int val2=val+val;
    val2 += val;
    val2 += val;
    val2 += val;
    b[5] = val2;
}

extern "C" void core_7_3() {
  int bounds = 2; // iter;

  // NOTE: odd iterations need locks reset externally when core is run again
  while (
      bounds >
      0) { // TODO: need to change this to start count at 0 so we do ping first
    if ((bounds & 0x1) == 0) {
      acquire(A_PING, LOCK_READ);
      acquire(B_PING, LOCK_WRITE);
      func(a_ping, b_ping);
      release(A_PING, LOCK_WRITE);
      release(B_PING, LOCK_READ);
    } else {
      acquire(A_PONG, LOCK_READ);
      acquire(B_PONG, LOCK_WRITE);
      func(a_pong, b_pong);
      release(A_PONG, LOCK_WRITE);
      release(B_PONG, LOCK_READ);
    }
    bounds--;
  }
}
