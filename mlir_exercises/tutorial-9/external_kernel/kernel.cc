//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
