//===- kernel.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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