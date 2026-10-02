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

void per_layer_up(bf16 *x, bf16 *proj_w_ping, bf16 *proj_w_pong, bf16 *y) {
  constexpr int x_prod_lock = FLM_GEMMA4_DECODE_PER_LAYER_UP_X_PROD_LOCK;
  constexpr int x_cons_lock = FLM_GEMMA4_DECODE_PER_LAYER_UP_X_CONS_LOCK;
  constexpr int proj_w_prod_lock =
      FLM_GEMMA4_DECODE_PER_LAYER_UP_PROJ_W_PROD_LOCK;
  constexpr int proj_w_cons_lock =
      FLM_GEMMA4_DECODE_PER_LAYER_UP_PROJ_W_CONS_LOCK;
  constexpr int y_prod_lock = FLM_GEMMA4_DECODE_PER_LAYER_UP_Y_PROD_LOCK;
  constexpr int y_cons_lock = FLM_GEMMA4_DECODE_PER_LAYER_UP_Y_CONS_LOCK;

  static PingPong proj_w_pingpong;

  alignas(aie::vector_decl_align) float y_acc[BF16_PROJ_M_BLOCK];
  bf16 *w_post_per_layer_input_norm = x;
  bf16 *layer_scale = x + MODEL_DIM;
  bf16 *per_layer_input = x + MODEL_DIM + 32;
  bf16 *residual_x = x + MODEL_DIM + 32 + PLI_D;
  bf16 *gate = x + MODEL_DIM + 32 + PLI_D + MODEL_DIM;

  _lock_acquire_p(x, x_cons_lock);
  _lock_acquire_p(y, y_prod_lock);

  _apply_gate(per_layer_input, gate);

  linear_proj<MODEL_DIM, PLI_D>(y, proj_w_ping, proj_w_pong, per_layer_input,
                                y_acc, proj_w_pingpong, proj_w_prod_lock,
                                proj_w_cons_lock);

  rms_norm<MODEL_DIM>(y, y, w_post_per_layer_input_norm);
  residual_add<MODEL_DIM>(y, residual_x, y);
  scale_vectorized<MODEL_DIM>(y, *layer_scale);

  _lock_release_p(y, y_cons_lock);
  _lock_release_p(x, x_prod_lock);
}
}