//===- llama_kv_proto.cc ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Stage 0 plumbing prototype for on-chip KV append. Validates the one
// genuine hardware unknown: a worker can consume a HOST-FILLED kv buffer,
// modify one cache slot in-array, and produce an updated buffer that is BOTH
// drained back to host AND consumed by a second worker (read-after-write).
//
// Per-head kv buffer layout (mirrors the real per-slot KV format):
//   [0..4]   position (int32)
//   [4..8]   pad
//   [8 .. 8+T*4]            k_slot_scales (T fp32)
//   [8+T*4 .. 8+T*4+T*HD]   k body (T*HD int8)
//   [.. + T*4]              v_slot_scales (T fp32)
//   [.. + T*HD]             v body
//
// kv_append: copy kv_in -> kv_out, then overwrite slot[pos] (k body, v body,
// and the two per-slot scale headers) with a deterministic pattern so the
// host can verify exactly. kv_read: extract slot[pos] (k body byte 0, the k
// scale) into a tiny out buffer to prove read-after-write ordering.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#ifndef LLAMA_KVP_HEAD_DIM
#define LLAMA_KVP_HEAD_DIM 64
#endif
#ifndef LLAMA_KVP_T
#define LLAMA_KVP_T 128
#endif

static constexpr int kHD = LLAMA_KVP_HEAD_DIM;
static constexpr int kT = LLAMA_KVP_T;
static constexpr int kPrefix = 8;
static constexpr int kScaleBytes = kT * 4;
static constexpr int kBodyBytes = kT * kHD;
static constexpr int kKHalf = kScaleBytes + kBodyBytes;
static constexpr int kPerHead = kPrefix + 2 * kKHalf;

extern "C" {

// Copy the whole per-head buffer, then write a deterministic pattern into
// slot[pos]: k body bytes = (int8)(pos+1), v body bytes = (int8)(pos+2),
// k scale = 0.0123f, v scale = 0.0456f.
void llama_kv_proto_append(int8_t *restrict kv_in, int8_t *restrict kv_out) {
  int32_t pos;
  memcpy(&pos, kv_in, 4);
  if (pos < 0)
    pos = 0;
  if (pos >= kT)
    pos = kT - 1;

  for (int i = 0; i < kPerHead; i++)
    kv_out[i] = kv_in[i];

  int8_t *k_scales = kv_out + kPrefix;
  int8_t *k_body = k_scales + kScaleBytes;
  int8_t *v_scales = k_body + kBodyBytes;
  int8_t *v_body = v_scales + kScaleBytes;

  for (int d = 0; d < kHD; d++) {
    k_body[pos * kHD + d] = (int8_t)(pos + 1);
    v_body[pos * kHD + d] = (int8_t)(pos + 2);
  }
  float ks = 0.0123f, vs = 0.0456f;
  memcpy(k_scales + pos * 4, &ks, 4);
  memcpy(v_scales + pos * 4, &vs, 4);
}

// Read-after-write check: extract slot[pos] proof bytes into out[0..12]:
//   out[0]   = k_body[pos*HD]   (should be pos+1)
//   out[1]   = v_body[pos*HD]   (should be pos+2)
//   out[4..8]  = k_scale[pos]   (should be 0.0123f)
//   out[8..12] = v_scale[pos]   (should be 0.0456f)
void llama_kv_proto_read(int8_t *restrict kv_in, int8_t *restrict out) {
  int32_t pos;
  memcpy(&pos, kv_in, 4);
  if (pos < 0)
    pos = 0;
  if (pos >= kT)
    pos = kT - 1;

  int8_t *k_scales = kv_in + kPrefix;
  int8_t *k_body = k_scales + kScaleBytes;
  int8_t *v_scales = k_body + kBodyBytes;
  int8_t *v_body = v_scales + kScaleBytes;

  for (int i = 0; i < 16; i++)
    out[i] = 0;
  out[0] = k_body[pos * kHD];
  out[1] = v_body[pos * kHD];
  memcpy(out + 4, k_scales + pos * 4, 4);
  memcpy(out + 8, v_scales + pos * 4, 4);
}

} // extern "C"
