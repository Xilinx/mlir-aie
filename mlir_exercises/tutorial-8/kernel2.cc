//===- kernel2.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel2(int32_t *restrict buf) {
  v16int16 v16 = get_scd_v16int16();
  buf[5] = ext_elem(v16, 0) + 100;
}

} // extern "C"