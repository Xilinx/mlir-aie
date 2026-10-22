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
