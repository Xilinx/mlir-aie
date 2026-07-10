//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

extern "C" {
void func(int32_t *a, int32_t *b);
}
