//===- decode_rope.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RoPE and the QK/V norm, templated on the head dims.
// FLM_GEMMA4_DECODE_ROPE_SWA=1 builds it for the sliding-window layers' head
// dim, 0 for the global-attention layers'.
#include "decode_geometry.h"
#include "decode_layout.h"
#include "rms_norm.h"
#include "utils.h"

template <int Dh>
void apply_rope_t(bf16 *restrict y, bf16 *restrict x, bf16 *restrict cos_val,
                  bf16 *restrict sin_val) {
  constexpr int vector_size = 16;
  const int Dh_2 = Dh / 2;
  constexpr int F = Dh / 2 / vector_size;
  bf16 *it_y_p1 = y;
  bf16 *it_y_p2 = y + Dh_2;
  bf16 *it_x_p1 = x;
  bf16 *it_x_p2 = x + Dh_2;
  bf16 *it_cos = cos_val;
  bf16 *it_sin = sin_val;

  for (int i = 0; i < F; i++) {
    aie::vector<bf16, vector_size> x1_vec = aie::load_v<vector_size>(it_x_p1);
    aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x_p2);
    aie::vector<bf16, vector_size> cos_vec = aie::load_v<vector_size>(it_cos);
    aie::vector<bf16, vector_size> sin_vec = aie::load_v<vector_size>(it_sin);
    aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
    aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
    C = aie::mac(C, x2_vec, neg_sin_vec);
    aie::accum<accfloat, vector_size> D = aie::mul(x1_vec, sin_vec);
    D = aie::mac(D, x2_vec, cos_vec);
    aie::store_v(it_y_p1, C.template to_vector<bf16>());
    aie::store_v(it_y_p2, D.template to_vector<bf16>());
    it_y_p1 += vector_size;
    it_y_p2 += vector_size;
    it_x_p1 += vector_size;
    it_x_p2 += vector_size;
    it_cos += vector_size;
    it_sin += vector_size;
  }
}

/// Norm then rotate one head. rope_w is [cos | sin] of Dh/2 each.
template <int Dh>
void _rotate_t(bf16 *restrict q, bf16 *restrict qkv, bf16 *restrict rope_w,
               bf16 *restrict norm_weight) {
  bf16 *it_rope = rope_w;
  bf16 *cos_val = it_rope;
  bf16 *sin_val = it_rope + Dh / 2;
  rms_norm<Dh>(qkv, qkv, norm_weight);
  apply_rope_t<Dh>(q, qkv, cos_val, sin_val);
}

#ifndef FLM_GEMMA4_DECODE_ROPE_SWA
#define FLM_GEMMA4_DECODE_ROPE_SWA 0
#endif

#if FLM_GEMMA4_DECODE_ROPE_SWA
constexpr int ROPE_DH = SWA_DH;
#else
constexpr int ROPE_DH = DH;
#endif

extern "C" {

// One head per call, into slot i of the output. rope_w is [cos | sin | q norm
// weight | k norm weight]; each entry point normalizes qkv in place first.
void rope_q(bf16 *restrict q, bf16 *restrict qkv, bf16 *restrict rope_w,
            int i) {
  _rotate_t<ROPE_DH>(q + i * ROPE_DH, qkv, rope_w, rope_w + ROPE_DH);
}

void rope_k(bf16 *restrict k, bf16 *restrict qkv, bf16 *restrict rope_w,
            int i) {
  _rotate_t<ROPE_DH>(k + i * ROPE_DH, qkv, rope_w, rope_w + 2 * ROPE_DH);
}

void rope_v(bf16 *restrict v, bf16 *restrict qkv, int i) {
  rms_norm_unweighted<ROPE_DH>(v + i * ROPE_DH, qkv);
}
}
