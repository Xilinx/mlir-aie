//===- test.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "kernel.h"

#define BUF_SIZE 1024
int main() {

  int32_t a[BUF_SIZE], b[BUF_SIZE], acc[BUF_SIZE], c[BUF_SIZE];
  /*
          for(int i=0; i<BUF_SIZE; i++) {
                  a[i] = i;
                  b[i] = 64+i;
              acc[i] = i+1;
                  c[i] = 0;
          }
  */
  extern_kernel(a, b, acc, c);
  return 0;
}
