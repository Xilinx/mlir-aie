//===- decode_bf16_proj_helper.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_bf16_proj.h"

#ifndef BF16_PROJ_HELPER
#error "decode_bf16_proj_helper.cc needs BF16_PROJ_HELPER defined"
#endif

#ifndef BF16_PROJ_SYMBOL
#error "decode_bf16_proj_helper.cc needs BF16_PROJ_SYMBOL defined"
#endif

extern "C" {

#if BF16_PROJ_HELPER == 0
void BF16_PROJ_SYMBOL(float *y_acc) {
  zero_256<float, BF16_PROJ_M_BLOCK>(y_acc);
}
#elif BF16_PROJ_HELPER == 1
void BF16_PROJ_SYMBOL(bf16 *w, bf16 *x, float *y_acc, int j, int x_offset) {
  _mvm_bf16_bf16<BF16_PROJ_M_BLOCK, BF16_PROJ_K_BLOCK>(
      w, x + x_offset + j * BF16_PROJ_K_BLOCK, y_acc);
}
#elif BF16_PROJ_HELPER == 2
void BF16_PROJ_SYMBOL(bf16 *y, float *y_acc, int i, int y_offset) {
  bf16 *y_using = y + y_offset + i * BF16_PROJ_M_BLOCK;
  aie::vector<float, BF16_PROJ_M_BLOCK> y_acc_vec =
      aie::load_v<BF16_PROJ_M_BLOCK>(y_acc);
  aie::accum<accfloat, BF16_PROJ_M_BLOCK> c_acc;
  c_acc.from_vector(y_acc_vec);
  aie::vector<bf16, BF16_PROJ_M_BLOCK> y_vec = c_acc.template to_vector<bf16>();
  aie::store_v(y_using, y_vec);
}
#else
#error "BF16_PROJ_HELPER must be 0 (zero), 1 (mvm), or 2 (store)"
#endif
}