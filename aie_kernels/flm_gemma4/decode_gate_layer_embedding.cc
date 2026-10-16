//===- decode_gate_layer_embedding.cc ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_bf16_proj.h"
#include "decode_geometry.h"
#include "decode_layout.h"
#include "lut_based_ops.h"
#include "utils.h"

void _activate(bf16 *x) {
  for (int i = 0; i < PLI_D / 16; i++) {
    aie::vector<bf16, 16> x_vec = aie::load_v<16>(x);
    aie::vector<bf16, 16> out_vec = getGeluBf16(x_vec);
    aie::store_v(x, out_vec);
    x += 16;
  }
}

extern "C" {

void gate_layer_embedding_begin(bf16 *x, bf16 *y) {
  copy_vectorized<bf16, MODEL_DIM>(y, x);
}

void gate_layer_embedding_finish(bf16 *y) { _activate(y + MODEL_DIM); }
}