//===- decode_glu_core.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_glu.cc"

extern "C" void glu_core(bf16 *x, bf16 *y) {
  event0();
  pseduo_glu<GLU_SLICE>(y, x);
  event1();
}
