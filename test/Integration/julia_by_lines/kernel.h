//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdint.h>

extern "C" {
void func(int32_t *a, float MinRe, float MaxRe, float MinIm, float MaxIm);
void do_line(int32_t *a, float MinRe, float StepRe, float Im, int size);
}
