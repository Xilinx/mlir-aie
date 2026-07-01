//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include <stdint.h>

extern "C" {
void func1(int *a, int *b);
void func2(int *a, int *b);
void func3(int *a, int *b);
void pass(int *a, int *b);

// void dequant_8x8(int16_t  *restrict input,
//              int16_t *restrict output);
}