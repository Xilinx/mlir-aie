//===- kernel13.cc ----------------------------------------------*- C++ -*-===//
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
#define EAST_ID_BASE 48

void func(int32_t *buf)
{
    acquire(EAST_ID_BASE+3,1);
    int tmp = buf[3];
    int val = tmp + tmp;
    val += tmp;
    val += tmp;
    val += tmp;
    v8acc48 v8acc;
    v8int16 v8;
    v8 = upd_elem(v8,0,val);
    v8acc = ups(v8,0);
    put_mcd(v8acc);
    release(EAST_ID_BASE+3,0);
}

int32_t buf[32];

int main()
{
    func(buf);
    //printf("test is %d\n",buf[8]);
}
