//===- llama_gqa_glue.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Glue kernels for the multi-head layer: q_requant (per-head self-calibrated
// int8 Q), qkv_combine (per-KV-head chunks for the on-chip KV append),
// sv_merge_selfcal (de-interleave the attention output) and af_concat
// (per-Q-head dequant + global self-calibrated requant), mirroring
// numpy_layer_mh's steps.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_GQA_HEAD_DIM
#define LLAMA_GQA_HEAD_DIM 64
#endif
#ifndef LLAMA_GQA_N_HEADS_Q
#define LLAMA_GQA_N_HEADS_Q 32
#endif
#ifndef LLAMA_GQA_N_HEADS_KV
#define LLAMA_GQA_N_HEADS_KV 8
#endif

static constexpr int kHD = LLAMA_GQA_HEAD_DIM;
static constexpr int kNHeadsQ = LLAMA_GQA_N_HEADS_Q;
static constexpr int kNHeadsKV = LLAMA_GQA_N_HEADS_KV;
static constexpr int kREP = kNHeadsQ / kNHeadsKV;
static constexpr int kQD = kNHeadsQ * kHD;             // 2048
static constexpr int kBodyChunk = kREP * kHD;          // 256
static constexpr int kTailChunk = kREP * 8;            // 32
static constexpr int kChunk = kBodyChunk + kTailChunk; // 288
static constexpr int kHalfKV = kNHeadsKV / 2;          // 4

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

// Software fp32 reciprocal (see llama_flowkv.cc); matches the other
// kernels' sw_recip so the numpy reference and device share the function.
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

// q_requant (Phase 2a self-cal): the q_proj gemm now emits fp32[kQD]; this
// stage computes each Q head's q_out_scale ON-CHIP (absmax over the head's
// 64-wide fp32 slice) and requants to int8, writing the per-head scale tail
// rope/flowkv expect: out[kQD + h*8 + 0..4] = q_scale_h, [+4..8] = pad
// (the old sv_inv_out slot, now self-calibrated by flowkv -- kept for the 8 B
// stride those kernels hardcode). out is int8[kQD + kNHeadsQ*8].
extern "C" void llama_q_requant(float *restrict q_fp, int8_t *restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  int8_t *tail = out + kQD;
  for (int h = 0; h < kNHeadsQ; h++) {
    float *src = q_fp + h * kHD;
    int8_t *dst = out + h * kHD;
    float absmax = 0.0f;
    for (int j = 0; j < kHD; j++) {
      float a = src[j] >= 0.0f ? src[j] : -src[j];
      if (a > absmax)
        absmax = a;
    }
    if (absmax < 1e-12f)
      absmax = 1e-12f;
    float q_scale = absmax * (1.0f / 127.0f);
    float q_inv = sw_recip(q_scale);
    for (int j = 0; j < kHD; j++)
      dst[j] = round_to_i8(src[j] * q_inv);
    memcpy(tail + h * 8 + 0, &q_scale, 4);
    int32_t zero = 0;
    memcpy(tail + h * 8 + 4, &zero, 4);
  }
  event1();
}

// Self-calibrating sv_merge: the selfcal flowkv writes each KV group's chunk
// as [REP*kHD body | REP*4 sv_out_scale] (272 B). The memtile join packs 4
// such chunks per half (in_lo/in_hi). This kernel DE-INTERLEAVES them into
//   out[0 .. kQD]                 = all 32 Q-head sv bodies, contiguous
//   out[kQD .. kQD + kNHeadsQ*4]  = all 32 per-Q-head sv_out_scales (fp32)
// so the downstream af_concat reads both bodies and sv_out_scales from this
// single buffer (no host af_scales fifo). Q-head order is preserved:
// KV group g holds Q heads [g*REP .. g*REP+REP).
static constexpr int kSvBody = kREP * kHD; // 256 (one KV group's body)
static constexpr int kSvTail = kREP * 4;   // 16  (one KV group's scales)
static constexpr int kSvChunk = kSvBody + kSvTail; // 272
static constexpr int kSvScalesOff = kQD;           // scales region start in out

extern "C" void llama_sv_merge_selfcal(int8_t *restrict in_lo,
                                       int8_t *restrict in_hi,
                                       int8_t *restrict out) {
  event0();
  int8_t *scales_out = out + kSvScalesOff;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *src = (g < kHalfKV) ? in_lo : in_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *chunk = src + local * kSvChunk;
    memcpy(out + g * kSvBody, chunk, kSvBody); // body -> contiguous
    memcpy(scales_out + g * kSvTail, chunk + kSvBody, kSvTail); // scales
  }
  event1();
}

// qkv_combine (on-chip KV append): builds per-head COMBINED chunks
// [q_chunk 288 B | k_fp 256 B | v_fp 256 B | cs 256 B] (1056 B/head) from
// rope's q output (288 B/head: 256 B body + 32 B scale tail) and a
// pre-joined kvcs buffer (kfp 2048 | vfp 2048 | cs 256 shared B, 8 heads
// of 64 fp32 each). Split into lo (heads 0..3) / hi (4..7) to keep memtile
// fanout <= 4.
static constexpr int kKfpHead = kHD * 4;     // 256 (64 fp32 for one head)
static constexpr int kCsBytes = 2 * kHD * 2; // 256 (cos+sin bf16)
static constexpr int kCombChunk = kChunk + 2 * kKfpHead + kCsBytes; // 1056

extern "C" void llama_qkv_combine(int8_t *restrict qr, int8_t *restrict kvcs,
                                  int8_t *restrict out_lo,
                                  int8_t *restrict out_hi) {
  event0();
  constexpr int kKfpRegion = kNHeadsKV * kKfpHead; // 8*256 = 2048
  int8_t *qr_body = qr;
  int8_t *qr_tail = qr + kQD;
  int8_t *kfp = kvcs;
  int8_t *vfp = kvcs + kKfpRegion;
  int8_t *cs = kvcs + 2 * kKfpRegion;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kCombChunk;
    // q_chunk: body (256) + tail (32)
    memcpy(dst, qr_body + g * kBodyChunk, kBodyChunk);
    memcpy(dst + kBodyChunk, qr_tail + g * kTailChunk, kTailChunk);
    // k_fp, v_fp, cs
    memcpy(dst + kChunk, kfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + kKfpHead, vfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + 2 * kKfpHead, cs, kCsBytes);
  }
  event1();
}

// af_concat, fully self-calibrating: sv_out_scales come from the merged sv
// buffer tail (written by sv_merge_selfcal), and the global o_act_scale is
// computed on-chip (absmax of af_fp = sv[h]*sv_out_scale[h] over Q_DIM) and
// written to af_out[kOutDim..kOutDim+4] (af_out is int8[kOutDim+8]) for the
// downstream o_proj. Input `sv` = [kQD bodies | kNHeadsQ*4 sv_out_scales].
extern "C" void llama_af_concat_selfcal2(int8_t *restrict sv,
                                         int8_t *restrict af_out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int kOutDim = kNHeadsQ * kHD;
  const float *sv_out_scales = reinterpret_cast<const float *>(sv + kOutDim);

  // Pass A: absmax of af_fp = sv[h]*sv_out_scale[h].
  float absmax = 0.0f;
  for (int h = 0; h < kNHeadsQ; h++) {
    float s = sv_out_scales[h];
    int8_t *src = sv + h * kHD;
    for (int j = 0; j < kHD; j++) {
      float v = (float)src[j] * s;
      float a = v >= 0.0f ? v : -v;
      if (a > absmax)
        absmax = a;
    }
  }
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float o_act_scale = absmax * (1.0f / 127.0f);
  float o_inv_act_scale = sw_recip(o_act_scale);

  // Pass B: requant.
  for (int h = 0; h < kNHeadsQ; h++) {
    float combined = sv_out_scales[h] * o_inv_act_scale;
    int8_t *src = sv + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int j = 0; j < kHD; j++) {
      dst[j] = round_to_i8((float)src[j] * combined);
    }
  }

  memcpy(af_out + kOutDim, &o_act_scale, 4);
  int32_t zero = 0;
  memcpy(af_out + kOutDim + 4, &zero, 4);
  event1();
}
