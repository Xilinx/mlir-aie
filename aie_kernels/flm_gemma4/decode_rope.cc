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

/// Norm then rotate one head into q; qkv is only read. rope_w is [cos | sin]
/// of Dh/2 each. Each half's normalized bf16 goes straight into the rotation.
template <int Dh>
void _rotate_t(bf16 *restrict q, bf16 *restrict qkv, bf16 *restrict rope_w,
               bf16 *restrict norm_weight) {
  constexpr int half = Dh / 2;
  static_assert(half % f32_lanes == 0);
  const Limbs<f32_lanes> s =
      bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(rms_scale<Dh>(qkv))));
  for (int i = 0; i < half; i += f32_lanes) {
    const bf16_vec x1 =
        weighted_norm(aie::load_v<f32_lanes>(qkv + i),
                      aie::load_v<f32_lanes>(norm_weight + i), s);
    const bf16_vec x2 =
        weighted_norm(aie::load_v<f32_lanes>(qkv + half + i),
                      aie::load_v<f32_lanes>(norm_weight + half + i), s);
    const bf16_vec cos = aie::load_v<f32_lanes>(rope_w + i);
    const bf16_vec sin = aie::load_v<f32_lanes>(rope_w + half + i);
    const f32_acc c = aie::msc(aie::mul(x1, cos), x2, sin);
    const f32_acc d = aie::mac(aie::mul(x1, sin), x2, cos);
    aie::store_v(q + i, c.to_vector<bfloat16>());
    aie::store_v(q + half + i, d.to_vector<bfloat16>());
  }
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
