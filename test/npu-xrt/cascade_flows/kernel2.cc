//===- kernel2.cc -------------------------------------------------*- C++
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

void extern_kernel2() {
  v32int32 v32 = get_scd_v32int32();
  v16int32 lo = upd_elem(extract_v16int32(v32, 0), 0, 114);
  put_mcd(insert(v32, 0, lo));
}

} // extern "C"