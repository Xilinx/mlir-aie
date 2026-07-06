//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>

extern "C" {
void do_line(int32_t *a, float MinRe, float StepRe, float Im, int size);
}
