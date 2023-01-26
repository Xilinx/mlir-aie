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
#include <stdio.h>

#define BUF_SIZE 256
int main() {

  int32_t buf[BUF_SIZE];

  extern_kernel(buf);
  printf("buf[3] = %d\n", buf[3]);
  return 0;
}
