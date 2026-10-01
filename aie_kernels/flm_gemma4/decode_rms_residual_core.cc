//===- decode_rms_residual_core.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A lock-free entry point around decode_rms_residual.cc's post-attention norm
// and residual add, for testing them in isolation.
#include "decode_rms_residual.cc"

// residual_add writes the sum back into its residual, so the wrapper adds a
// copy and leaves the input buffer as the DMA filled it. File scope: a stack
// frame cannot hold MODEL_DIM bf16.
alignas(aie::vector_decl_align) static bf16 rms_core_residual[MODEL_DIM];

// y = rms_norm(x, w) + residual, all MODEL_DIM bf16.
extern "C" void rms_residual_core(bf16 *restrict x, bf16 *restrict w,
                                  bf16 *restrict residual, bf16 *y) {
  event0();
  copy_vectorized<bf16, MODEL_DIM>(rms_core_residual, residual);
  rms_norm<MODEL_DIM>(y, x, w);
  residual_add<MODEL_DIM>(y, rms_core_residual, y);
  event1();
}
