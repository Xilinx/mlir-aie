//===- kernel23.cc ----------------------------------------------*- C++ -*-===//
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

#define EAST_ID_BASE 48

//void func(int32_t *buf);

void func(int32_t *buf)
{
    acquire(EAST_ID_BASE+7,0);
    int tmp = ext_elem(srs(get_scd(),0),0);
    int val = tmp + tmp;
    val += tmp;
    val += tmp;
    val += tmp;
    buf[5] = val;
    release(EAST_ID_BASE+7,1);
}

int32_t buf[32];

int main()
{
    func(buf);
    //printf("test is %d\n",buf[8]);
}
