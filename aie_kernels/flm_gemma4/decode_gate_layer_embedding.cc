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

void gate_layer_embedding(bf16 *x, bf16 *proj_w_ping, bf16 *proj_w_pong,
                          bf16 *y) {
  constexpr int proj_w_prod_lock =
      FLM_GEMMA4_DECODE_GATE_LAYER_EMBEDDING_PROJ_W_PROD_LOCK;
  constexpr int proj_w_cons_lock =
      FLM_GEMMA4_DECODE_GATE_LAYER_EMBEDDING_PROJ_W_CONS_LOCK;
  constexpr int final_x_prod_lock =
      FLM_GEMMA4_DECODE_GATE_LAYER_EMBEDDING_FINAL_X_PROD_LOCK;
  constexpr int final_x_cons_lock =
      FLM_GEMMA4_DECODE_GATE_LAYER_EMBEDDING_FINAL_X_CONS_LOCK;

  static PingPong proj_w_pingpong;
  alignas(aie::vector_decl_align) float y_acc[BF16_PROJ_M_BLOCK];
  _left_lock_acquire_p(x, final_x_cons_lock);

  copy_vectorized<bf16, MODEL_DIM>(y, x);
  _left_lock_release_p(x, final_x_prod_lock);
  linear_proj<PLI_D, MODEL_DIM>(y + MODEL_DIM, proj_w_ping, proj_w_pong, y,
                                y_acc, proj_w_pingpong, proj_w_prod_lock,
                                proj_w_cons_lock);
  _activate(y + MODEL_DIM);
}
}