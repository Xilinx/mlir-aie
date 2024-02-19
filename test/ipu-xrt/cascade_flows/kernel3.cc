//===- kernel3.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel3(int32_t *restrict buf, int N) {
  v32int32 v32 = get_scd_v32int32();
  for (int i = 0; i < N; i++) {
    if (i == 5) {
      buf[i] = ext_elem(v32, 0) + 100;
    } else {
      buf[i] = 0;
    }
  }
}

} // extern "C"