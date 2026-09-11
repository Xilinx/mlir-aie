//===- kernel1.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel1() {
  v16int32 lo = upd_elem(undef_v16int32(), 0, 14);
  put_mcd(concat(lo, undef_v16int32()));
}

} // extern "C"