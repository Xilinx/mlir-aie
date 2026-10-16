//===- decode_per_layer_up_core.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_per_layer_up.cc"

extern "C" void bf16_proj_block_core(bf16 *w, bf16 *x, float *y) {
  event0();
  // The accumulator must share a frame with the inlined MVM for the Peano
  // loop-back-edge fix described in decode_bf16_proj.h.
  alignas(aie::vector_decl_align) float y_acc[BF16_PROJ_M_BLOCK];
  zero_256<float, BF16_PROJ_M_BLOCK>(y_acc);
  _mvm_bf16_bf16<BF16_PROJ_M_BLOCK, BF16_PROJ_K_BLOCK>(w, x, y_acc);
  copy_vectorized<float, BF16_PROJ_M_BLOCK>(y, y_acc);
  event1();
}
