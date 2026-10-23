//===- llama_gqa_glue.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Glue kernels for the multi-head layer: comb_rope_rows (per-KV-head chunks
// for the attention and the on-chip KV append), sv_merge (join the attention
// output) and af_quant (the o projection's input, quantized once), mirroring
// llama_oracle's steps.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_kv_row.h"
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
static constexpr int kQD = kNHeadsQ * kHD;
static constexpr int kBodyChunk = kREP * kHD;
static constexpr int kTailChunk = kREP * 8;
static constexpr int kChunk = kBodyChunk + kTailChunk;
static constexpr int kHalfKV = kNHeadsKV / 2;

// One 512-bit fp32 register; wider vectors made Peano spill to 3 KB frames.
static constexpr int kVec = 16;
static_assert(kHD % kVec == 0, "LLAMA_GQA_HEAD_DIM must be a multiple of 16");

static inline float bits_to_float(int32_t bits) {
  float f = __builtin_bit_cast(float, bits);
  return f;
}

// Each KV group's chunk is its REP Q heads' sv, kREP*kHD fp32; the memtile
// join packs 4 chunks per half, KV groups 0-3 in lo and 4-7 in hi. KV group
// g holds Q heads [g*REP, g*REP + REP), so lo then hi is every Q head's sv
// in order, which af_quant reads from one buffer.
static constexpr int kSvHalfBytes = kHalfKV * kREP * kHD * 4;

extern "C" void llama_sv_merge(int8_t *restrict in_lo, int8_t *restrict in_hi,
                               int8_t *restrict out) {
  event0();
  copy_words<kSvHalfBytes>(out, in_lo);
  copy_words<kSvHalfBytes>(out + kSvHalfBytes, in_hi);
  event1();
}

static constexpr int kKfpHead = kHD * 4;

// The attention's per-head chunks [q_chunk 288 B | row 136 B], the row being
// the head's new K/V cache row, [k int8 | v int8 | k_scale f32 | v_scale f32]
// (llama_kv_row.h).
static constexpr int kRowBytes = 2 * kHD + 8;
static constexpr int kRowChunk = kChunk + kRowBytes;

// comb_rope_rows: the GEMV array's per-KV-group chunks
// [roped q body 256 B | q tail 32 B | k fp32 x64 | v fp32 x64 | cs 256 B]
// (llama_gemv_array_qkv_comb_cs), all 8 groups in comb, to the attention's
// [q 288 B | row 136 B] chunks, lo (groups 0-3) and hi (4-7), each half
// followed by its 4 rows again, for the cache write-back.
static constexpr int kArrayChunk = kChunk + 3 * kKfpHead;

extern "C" void llama_comb_rope_rows(int8_t *restrict comb,
                                     int8_t *restrict out_lo,
                                     int8_t *restrict out_hi) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kRowChunk;
    const int8_t *src = comb + g * kArrayChunk;
    copy_words<kChunk>(dst, src);
    alignas(64) int8_t row[kRowBytes];
    llama_kv_row_quant<kHD>(src + kChunk, row);
    copy_words<kRowBytes>(dst + kChunk, row);
    copy_words<kRowBytes>(base + kHalfKV * kRowChunk + local * kRowBytes, row);
  }
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// The o projection's input: every Q head's sv (fp32[kQD], as
// llama_sv_merge joins it) quantized once to int8 with one scale, its
// absmax / 127, at af_out[kQD] (af_out is int8[kQD + 8]).
extern "C" void llama_af_quant(int8_t *restrict sv, int8_t *restrict af_out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *x = reinterpret_cast<const float *>(sv);

  // A non-NaN float's |v| bits order as an int32, so the absmax is an
  // integer max.
  ::aie::vector<int32, kVec> mag = ::aie::zeros<int32, kVec>();
  for (int c = 0; c < kQD / kVec; c++)
    mag = ::aie::max(
        mag, ::aie::bit_and(
                 int32(0x7fffffff),
                 ::aie::load_v<kVec>(x + c * kVec).template cast_to<int32>()));
  float absmax = bits_to_float(::aie::reduce_max(mag));
  // absmax is always a magnitude (sign bit clear, from the bit_and with
  // 0x7fffffff above), so its bit pattern orders the same as its value: an
  // unsigned int compare avoids __ltsf2.
  if ((uint32_t)__builtin_bit_cast(int32_t, absmax) <
      (uint32_t)__builtin_bit_cast(int32_t, 1e-12f))
    absmax = 1e-12f;
  float o_act_scale = scalar_mul_rne(absmax, 1.0f / 127.0f);
  ::aie::vector<float, kVec> inv_v =
      ::aie::broadcast<float, kVec>(recip_rne(o_act_scale));
  for (int c = 0; c < kQD / kVec; c++)
    ::aie::store_v(
        af_out + c * kVec,
        vec_round_to_i8(vec_mul_rne(::aie::load_v<kVec>(x + c * kVec), inv_v)));

  __builtin_memcpy(af_out + kQD, &o_act_scale, 4);
  int32_t zero = 0;
  __builtin_memcpy(af_out + kQD + 4, &zero, 4);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}
