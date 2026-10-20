//===- llama_kv_row.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// One token's K/V cache row from its fp32 k and v: rope_k, then K and V each
// quantized to int8 with its own absmax / 127 scale (numpy_layer_mh's append).
//===----------------------------------------------------------------------===//

#ifndef LLAMA_KV_ROW_H
#define LLAMA_KV_ROW_H

#include <stdint.h>
#include <string.h>

static inline int8_t llama_kv_row_round(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > 127)
    r = 127;
  if (r < -128)
    r = -128;
  return (int8_t)r;
}

// Pure IEEE fp32 reciprocal (Peano AIE2P `/` is a HW approximation, Bug 1).
static inline float llama_kv_row_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

// kvfp_packed is [k fp32[kHD] | v fp32[kHD] | cos bf16[kHD] | sin bf16[kHD]];
// writes kHD int8 to each of k_out and v_out, and the two scales.
template <int kHD>
static inline void
llama_kv_row_quant(const int8_t *restrict kvfp_packed, int8_t *restrict k_out,
                   int8_t *restrict v_out, float *restrict k_scale,
                   float *restrict v_scale) {
  constexpr int kHalf = kHD / 2;
  const float *k_fp = reinterpret_cast<const float *>(kvfp_packed);
  const float *v_fp = reinterpret_cast<const float *>(kvfp_packed + kHD * 4);
  const bfloat16 *cos =
      reinterpret_cast<const bfloat16 *>(kvfp_packed + kHD * 8);
  const bfloat16 *sin = cos + kHD;

  // 1) rope_k (half-split). Snapshot halves to locals so the in-place write
  // can't alias (the rotate-half aliasing bug fixed in the numpy ref).
  float k_rope[kHD];
  for (int i = 0; i < kHalf; i++) {
    float x1 = k_fp[i];
    float x2 = k_fp[kHalf + i];
    float c1 = (float)cos[i];
    float s1 = (float)sin[i];
    float c2 = (float)cos[kHalf + i];
    float s2 = (float)sin[kHalf + i];
    k_rope[i] = x1 * c1 - x2 * s1;
    k_rope[kHalf + i] = x2 * c2 + x1 * s2;
  }

  // 2) per-slot dynamic scale for K (absmax / 127, 1e-12 floor).
  float k_absmax = 0.0f;
  for (int i = 0; i < kHD; i++) {
    float a = k_rope[i] >= 0.0f ? k_rope[i] : -k_rope[i];
    if (a > k_absmax)
      k_absmax = a;
  }
  if (k_absmax < 1e-12f)
    k_absmax = 1e-12f;
  float ks = k_absmax * (1.0f / 127.0f);
  float k_inv = llama_kv_row_recip(ks);

  // 3) per-slot dynamic scale for V (no rope).
  float v_absmax = 0.0f;
  for (int i = 0; i < kHD; i++) {
    float a = v_fp[i] >= 0.0f ? v_fp[i] : -v_fp[i];
    if (a > v_absmax)
      v_absmax = a;
  }
  if (v_absmax < 1e-12f)
    v_absmax = 1e-12f;
  float vs = v_absmax * (1.0f / 127.0f);
  float v_inv = llama_kv_row_recip(vs);

  // 4) requant.
  for (int i = 0; i < kHD; i++) {
    k_out[i] = llama_kv_row_round(k_rope[i] * k_inv);
    v_out[i] = llama_kv_row_round(v_fp[i] * v_inv);
  }
  *k_scale = ks;
  *v_scale = vs;
}

#endif // LLAMA_KV_ROW_H
