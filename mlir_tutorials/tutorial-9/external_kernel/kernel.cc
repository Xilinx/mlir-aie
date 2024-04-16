//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 1
#define NOCPP

#define h1 32
#define w1 32
#define w2 32

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel(int32_t *restrict buf) { buf[3] = 14; }

} // extern "C"
