//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

void func(int32_t *a, int32_t *b, int32_t size)
{
    int new_size = 256;
    for (int32_t i = 0; i < 256; i ++){
        b[i] = a[i];
    }
}