//===- decode_rms_residual_core.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_rms_residual.cc"

// y = rms_norm(x, w) + residual, all MODEL_DIM bf16.
extern "C" void rms_residual_core(bf16 *restrict x, bf16 *restrict w,
                                  bf16 *restrict residual, bf16 *restrict y) {
  event0();
  rms_norm_add<MODEL_DIM>(y, x, w, residual);
  event1();
}
