//===- llama_kv_append.cc -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// On-chip KV append (one KV head). Computes the current token's K/V cache
// slot ON-DEVICE and writes it into the head's persistent cache buffer at
// `position`, so the device owns the cache (no host-side k_proj/v_proj/
// rope_k/quant). Bit-exact mirror of numpy_layer_mh.py:426-465 (the
// `position=` path) for a single head:
//
//   k_rope = rotate_half(k_fp) * cos/sin    (Llama half-split convention)
//   ks = max(|k_rope|, 1e-12) / 127          (per-slot dynamic scale)
//   cache_k[pos] = round(k_rope / ks)        (1/ks via sw_recip)
//   k_slot_scales[pos] = ks
//   (v: no rope; vs = max(|v_fp|,1e-12)/127; cache_v[pos] = round(v_fp/vs))
//
// Inputs are this head's k_fp/v_fp (fp32, HEAD_DIM each, from k_proj/v_proj)
// and cos/sin (bf16, HEAD_DIM each = two copies of the half). The cache
// buffer is the per-head KV slot in the flowkv_mh_kvc layout:
//   [0..4]   position (int32)   -- written slot index = position
//   [4..8]   pad
//   [8 .. 8+T*4]            k_slot_scales (T fp32)
//   [8+T*4 .. 8+T*4+T*HD]   k body (T*HD int8)
//   [.. + T*4]              v_slot_scales (T fp32)
//   [.. + T*HD]             v body
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#ifndef LLAMA_KVA_HEAD_DIM
#define LLAMA_KVA_HEAD_DIM 64
#endif
#ifndef LLAMA_KVA_T
#define LLAMA_KVA_T 128
#endif

static constexpr int kHD = LLAMA_KVA_HEAD_DIM;
static constexpr int kT = LLAMA_KVA_T;
static constexpr int kHalf = kHD / 2;
static constexpr int kPrefix = 8;
static constexpr int kScaleBytes = kT * 4;
static constexpr int kBodyBytes = kT * kHD;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

// Pure IEEE fp32 reciprocal (Peano AIE2P `/` is a HW approximation, Bug 1).
static inline float sw_recip(float a) {
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

extern "C" {

// Co-located variant: the attn tile holds one COMBINED per-head chunk
//   [q_chunk 288 B | k_fp 256 B | v_fp 256 B | cs 256 B]
// The append step reads [k_fp|v_fp|cs] at offset 288 (= kQChunkBytes). This
// wrapper lets the same tile feed flowkv_mh_kvc(combined, ...) (which reads
// only the first 288 B) and append(combined+288, ...) without pointer math
// in the IRON worker body. kQChunkBytes = REP*HEAD_DIM (256) + REP*8 (32).
void llama_kv_append_head(int8_t *restrict kvfp_packed, int8_t *restrict kv_in,
                          int8_t *restrict kv_out);
void llama_kv_append_combined(int8_t *restrict combined, int8_t *restrict kv_in,
                              int8_t *restrict kv_out) {
  constexpr int kREP = 4; // N_HEADS_Q / N_HEADS_KV for Llama 3.2 1B
  constexpr int kQChunkBytes = kREP * kHD + kREP * 8; // 288
  llama_kv_append_head(combined + kQChunkBytes, kv_in, kv_out);
}

// One KV head. kvfp_packed is ONE input fifo packing [k_fp fp32[HEAD_DIM] |
// v_fp fp32[HEAD_DIM] | cs bf16[2*HEAD_DIM]] -- packed to stay within the
// 2-in/2-out compute-tile DMA budget (k_fp + v_fp + cs as 3 separate inputs
// plus kv_in would be 4 inputs, exceeding the limit). kv_in: host-supplied
// per-head cache (layout above; position in its [0..4] prefix). kv_out: the
// updated cache (copy of kv_in with slot[pos] overwritten) -- fans to
// flowkv_mh AND the host drain (device-owned cache). Separate in/out because
// IRON object-fifos give distinct input/output buffer instances (Stage 0
// proved this copy-in->out pattern).
void llama_kv_append_head(int8_t *restrict kvfp_packed, int8_t *restrict kv_in,
                          int8_t *restrict kv_out) {
  constexpr int kPerHead = kPrefix + 2 * (kScaleBytes + kBodyBytes);
  const float *k_fp = reinterpret_cast<const float *>(kvfp_packed);
  const float *v_fp = reinterpret_cast<const float *>(kvfp_packed + kHD * 4);
  const bfloat16 *cs_packed =
      reinterpret_cast<const bfloat16 *>(kvfp_packed + kHD * 8);

  // The cache prefix [0..4] holds T_used (the flowkv_mh_kvc contract: number
  // of valid cached slots including the one we're about to write). The append
  // slot index is therefore position = T_used - 1.
  int32_t t_used;
  memcpy(&t_used, kv_in, 4);
  int32_t pos = t_used - 1;
  if (pos < 0)
    pos = 0;
  if (pos >= kT)
    pos = kT - 1;

  for (int i = 0; i < kPerHead; i++)
    kv_out[i] = kv_in[i];

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;

  int8_t *k_slot_scales = kv_out + kPrefix;
  int8_t *k_body = k_slot_scales + kScaleBytes;
  int8_t *v_slot_scales = k_body + kBodyBytes;
  int8_t *v_body = v_slot_scales + kScaleBytes;

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
  float k_inv = sw_recip(ks);

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
  float v_inv = sw_recip(vs);

  // 4) requant + write slot[pos] (body + per-slot scale header).
  for (int i = 0; i < kHD; i++) {
    k_body[pos * kHD + i] = round_to_i8(k_rope[i] * k_inv);
    v_body[pos * kHD + i] = round_to_i8(v_fp[i] * v_inv);
  }
  memcpy(k_slot_scales + pos * 4, &ks, 4);
  memcpy(v_slot_scales + pos * 4, &vs, 4);
}

} // extern "C"
