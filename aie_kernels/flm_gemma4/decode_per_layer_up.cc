//===- decode_per_layer_up.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_bf16_proj.h"
#include "decode_geometry.h"
#include "decode_layout.h"
#include "rms_norm.h"
#include "utils.h"

void _apply_gate(bf16 *y, const bf16 *gate) {
  for (int i = 0; i < PLI_D / 16; i++) {
    aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
    aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate);
    aie::vector<bf16, 16> out_vec = aie::mul(y_vec, gate_vec);
    aie::store_v(y, out_vec);
    y += 16;
    gate += 16;
  }
}

extern "C" {

void per_layer_up_begin(bf16 *x) {
  bf16 *per_layer_input = x + MODEL_DIM + 32;
  bf16 *gate = x + MODEL_DIM + 32 + PLI_D + MODEL_DIM;

  _apply_gate(per_layer_input, gate);
}

void per_layer_up_finish(bf16 *x, bf16 *y) {
  bf16 *w_post_per_layer_input_norm = x;
  bf16 *layer_scale = x + MODEL_DIM;
  bf16 *residual_x = x + MODEL_DIM + 32 + PLI_D;
  rms_norm<MODEL_DIM>(y, y, w_post_per_layer_input_norm);
  residual_add<MODEL_DIM>(y, residual_x, y);
  scale_vectorized<MODEL_DIM>(y, *layer_scale);
}
}