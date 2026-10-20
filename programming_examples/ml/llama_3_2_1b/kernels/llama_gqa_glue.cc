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

#include "llama_kv_row.h"
#include "llama_rope_heads.h"
#include "llama_vec_fp.h"

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

// One 512-bit fp32 register; wider vectors made Peano spill to 3 KB frames.
static constexpr int kVec = 16;
static_assert(kHD % kVec == 0, "LLAMA_GQA_HEAD_DIM must be a multiple of 16");

// See vec_sw_recip in llama_vec_fp.h; this is its scalar form.
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

static inline float bits_to_float(int32_t bits) {
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// Each KV group's chunk is [REP*kHD body | REP*4 sv_out_scale]; the memtile
// join packs 4 chunks per half. De-interleave them into [kQD bodies |
// kNHeadsQ*4 scales] so af_concat reads both from one buffer. KV group g
// holds Q heads [g*REP, g*REP + REP), so Q-head order is preserved.
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

static constexpr int kKfpHead = kHD * 4;     // 256 (64 fp32 for one head)
static constexpr int kCsBytes = 2 * kHD * 2; // 256 (cos+sin bf16)

// The attention's per-head chunks [q_chunk 288 B | row 136 B], the row being
// the head's new K/V cache row, [k int8 | v int8 | k_scale f32 | v_scale f32]
// (llama_kv_row.h).
static constexpr int kRowBytes = 2 * kHD + 8;        // 136
static constexpr int kRowChunk = kChunk + kRowBytes; // 424

// comb_rope_rows (the GEMV array decode): the array's per-KV-group chunks
// [q body 256 B | q tail 32 B | k fp32 x64 | v fp32 x64 | cs 256 B]
// (llama_gemv_array_qkv_comb_cs), all 8 groups in comb, to the attention's
// [q roped 288 B | row 136 B] chunks, lo (groups 0-3) and hi (4-7), each
// half followed by its 4 rows again, for the cache write-back.
static constexpr int kArrayChunk = kChunk + 3 * kKfpHead; // 1056

extern "C" void llama_comb_rope_rows(int8_t *restrict comb,
                                     int8_t *restrict out_lo,
                                     int8_t *restrict out_hi) {
  event0();
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kRowChunk;
    const int8_t *src = comb + g * kArrayChunk;
    const int8_t *kvfp = src + kChunk;
    // Rope's vector loads and stores need 64-byte alignment, which neither
    // the chunks here (1056 B apart) nor the outputs (424 B apart) keep.
    alignas(64) int8_t q[kChunk];
    alignas(64) bfloat16 cs[2 * kHD];
    alignas(64) int8_t roped[kChunk];
    memcpy(q, src, kChunk);
    memcpy(cs, kvfp + 2 * kKfpHead, kCsBytes);
    llama_rope_heads<kHD, kREP>(q, cs, roped);
    memcpy(dst, roped, kChunk);
    int8_t *row = dst + kChunk;
    float ks, vs;
    llama_kv_row_quant<kHD>(kvfp, row, row + kHD, &ks, &vs);
    memcpy(row + 2 * kHD, &ks, 4);
    memcpy(row + 2 * kHD + 4, &vs, 4);
    memcpy(base + kHalfKV * kRowChunk + local * kRowBytes, row, kRowBytes);
  }
  event1();
}

// Writes o_proj's activation scale to af_out[kOutDim] (af_out is
// int8[kOutDim + 8]). For the layout of `sv`, see llama_sv_merge_selfcal.
extern "C" void llama_af_concat_selfcal2(int8_t *restrict sv,
                                         int8_t *restrict af_out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  constexpr int kOutDim = kNHeadsQ * kHD;
  const float *sv_out_scales = reinterpret_cast<const float *>(sv + kOutDim);

  // Pass A: absmax of af_fp = sv[h]*sv_out_scale[h]. A non-NaN float's |v|
  // bits order as an int32, so the absmax is an integer max.
  ::aie::vector<int32, kVec> mag = ::aie::zeros<int32, kVec>();
  for (int h = 0; h < kNHeadsQ; h++) {
    ::aie::vector<float, kVec> s_v =
        ::aie::broadcast<float, kVec>(sv_out_scales[h]);
    int8_t *src = sv + h * kHD;
    for (int c = 0; c < kHD / kVec; c++)
      mag = ::aie::max(
          mag,
          ::aie::bit_and(int32(0x7fffffff),
                         vec_mul_rne(::aie::to_float<float>(
                                         ::aie::load_v<kVec>(src + c * kVec)),
                                     s_v)
                             .template cast_to<int32>()));
  }
  float absmax = bits_to_float(::aie::reduce_max(mag));
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float o_act_scale = absmax * (1.0f / 127.0f);
  float o_inv_act_scale = sw_recip(o_act_scale);

  // Pass B: requant with combined = sv_out_scale[h] * o_inv_act_scale.
  alignas(64) float combined[kNHeadsQ];
  for (int g = 0; g < kNHeadsQ / kVec; g++)
    ::aie::store_v(combined + g * kVec,
                   vec_mul_rne(::aie::load_v<kVec>(sv_out_scales + g * kVec),
                               ::aie::broadcast<float, kVec>(o_inv_act_scale)));
  for (int h = 0; h < kNHeadsQ; h++) {
    ::aie::vector<float, kVec> combined_v =
        ::aie::broadcast<float, kVec>(combined[h]);
    int8_t *src = sv + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int c = 0; c < kHD / kVec; c++)
      ::aie::store_v(
          dst + c * kVec,
          vec_round_to_i8(vec_mul_rne(
              ::aie::to_float<float>(::aie::load_v<kVec>(src + c * kVec)),
              combined_v)));
  }

  memcpy(af_out + kOutDim, &o_act_scale, 4);
  int32_t zero = 0;
  memcpy(af_out + kOutDim + 4, &zero, 4);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}
