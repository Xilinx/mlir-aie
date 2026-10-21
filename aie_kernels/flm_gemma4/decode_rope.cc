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

/// The kernel body. Dq/Dk/Dv are the total q/k/v widths; Dh the head dim.
template <int Dh, int Dq, int Dk, int Dv>
void rope_body(bf16 *restrict q, bf16 *restrict k, bf16 *restrict v,
               bf16 *restrict qkv_ping, bf16 *restrict qkv_pong,
               bf16 *restrict rope_w, int *SKIP_KV) {
  constexpr int qkv_prod_lock = FLM_GEMMA4_DECODE_ROPE_QKV_PROD_LOCK;
  constexpr int qkv_cons_lock = FLM_GEMMA4_DECODE_ROPE_QKV_CONS_LOCK;
  constexpr int k_prod_lock = FLM_GEMMA4_DECODE_ROPE_K_PROD_LOCK;
  constexpr int k_cons_lock = FLM_GEMMA4_DECODE_ROPE_K_CONS_LOCK;
  constexpr int v_prod_lock = FLM_GEMMA4_DECODE_ROPE_V_PROD_LOCK;
  constexpr int v_cons_lock = FLM_GEMMA4_DECODE_ROPE_V_CONS_LOCK;
  constexpr int rope_prod_lock = FLM_GEMMA4_DECODE_ROPE_ROPE_PROD_LOCK;
  constexpr int rope_cons_lock = FLM_GEMMA4_DECODE_ROPE_ROPE_CONS_LOCK;
  static PingPong qkv_pingpong;
  _lock_acquire_p(rope_w, rope_cons_lock);

  for (int i = 0; i < Dq / Dh; i++) {
    bf16 *qkv_using = qkv_pingpong.acquire(qkv_ping, qkv_pong, qkv_cons_lock);
    _rotate_t<Dh>(q + i * Dh, qkv_using, rope_w, rope_w + Dh);
    _lock_release_p(qkv_using, qkv_prod_lock);
  }

  if (SKIP_KV[0] == 0) {
    _lock_acquire_p(k, k_prod_lock, 1);
    for (int i = 0; i < Dk / Dh; i++) {
      bf16 *qkv_using = qkv_pingpong.acquire(qkv_ping, qkv_pong, qkv_cons_lock);
      _rotate_t<Dh>(k + i * Dh, qkv_using, rope_w, rope_w + Dh + Dh);
      _lock_release_p(qkv_using, qkv_prod_lock);
    }
    _lock_release_p(k, k_cons_lock, 1);

    _lock_acquire_p(v, v_prod_lock, 1);
    for (int i = 0; i < Dv / Dh; i++) {
      bf16 *qkv_using = qkv_pingpong.acquire(qkv_ping, qkv_pong, qkv_cons_lock);
      rms_norm_unweighted<Dh>(v + i * Dh, qkv_using);
      _lock_release_p(qkv_using, qkv_prod_lock);
    }
    _lock_release_p(v, v_cons_lock, 1);
  }

  _lock_release_p(rope_w, rope_prod_lock);
}

#ifndef FLM_GEMMA4_DECODE_ROPE_SWA
#define FLM_GEMMA4_DECODE_ROPE_SWA 0
#endif

extern "C" {

void rope(bf16 *restrict q, bf16 *restrict k, bf16 *restrict v,
          bf16 *restrict qkv_ping, bf16 *restrict qkv_pong,
          bf16 *restrict rope_w, int *SKIP_KV) {
#if FLM_GEMMA4_DECODE_ROPE_SWA
  rope_body<SWA_DH, SWA_DQ, SWA_DK, SWA_DV>(q, k, v, qkv_ping, qkv_pong, rope_w,
                                            SKIP_KV);
#else
  rope_body<DH, DQ, DK, DV>(q, k, v, qkv_ping, qkv_pong, rope_w, SKIP_KV);
#endif
}
}
