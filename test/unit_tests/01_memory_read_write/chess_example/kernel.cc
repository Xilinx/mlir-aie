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

void func(int32_t *buf)
{
    int val=7;
    val = val+val;
    chess_report(val);
    buf[3] = val;
    val = 8;
    buf[5] = val;
    val = buf[3];
    buf[9] = val;
}

int32_t buf[32];

int main()
{
    func(buf);
    //printf("test is %d\n",buf[8]);
}
