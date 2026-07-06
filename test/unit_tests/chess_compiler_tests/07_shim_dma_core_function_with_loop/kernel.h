//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include <stdint.h>

extern "C" {
void func(int *a, int *b, int size);
}