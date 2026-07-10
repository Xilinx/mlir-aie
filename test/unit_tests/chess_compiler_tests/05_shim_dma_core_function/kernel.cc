//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int32_t *a, int32_t *b)
{
    int val=a[3];
    int val2=val+val;
    val2 += val;
    val2 += val;
    val2 += val;
    b[5] = val2;
}