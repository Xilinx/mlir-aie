//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#include <stdlib.h>

extern "C" {

void bfp16_passthrough_scalar(int8_t *in, int8_t *out) {
  for (int i = 0; i < 128 * 1.125; i++) {
    out[i] = in[i];
  }
}

void bfp16_passthrough_vectorized(v128bfp16ebs8 *in, v128bfp16ebs8 *out) {
  // --- Passthrough ---
  *out = *in;
}

} // extern "C"
