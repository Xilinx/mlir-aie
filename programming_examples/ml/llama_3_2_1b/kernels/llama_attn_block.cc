//===- llama_attn_block.cc ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Block-wise GQA attention for one KV head's kREP Q heads over a cache of
// kTB-row blocks, a row being [k int8[kHD] | v int8[kHD] | k_scale | v_scale].
// Each block is attended as llama_flowkv_mh attends a whole cache, then
// folded into a running (max, sum, normalized sv) state; products round once
// as the scalar `*` does and sums are vadd.f, so numpy_attn_block.py mirrors
// it exactly. One block gives llama_flowkv_mh_selfcal's output.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "exp_lut.h"
#include "llama_vec_fp.h"

#ifndef LLAMA_ATTN_HEAD_DIM
#define LLAMA_ATTN_HEAD_DIM 64
#endif
#ifndef LLAMA_ATTN_TB
#define LLAMA_ATTN_TB 64
#endif
#ifndef LLAMA_ATTN_REP
#define LLAMA_ATTN_REP 4
#endif
#ifndef LLAMA_ATTN_EXP_QUANT_SCALE
#define LLAMA_ATTN_EXP_QUANT_SCALE 0.05f
#endif

static constexpr int kHD = LLAMA_ATTN_HEAD_DIM;
static constexpr int kTB = LLAMA_ATTN_TB;
static constexpr int kREP = LLAMA_ATTN_REP;
static constexpr int kLanes = 16;
static constexpr int kGroups = kTB / kLanes;
static constexpr int kSlices = kHD / kLanes;
static_assert(kSlices == 4, "the sv loop keeps four named accumulators");
static constexpr float kInvExpQuantScale = 1.0f / LLAMA_ATTN_EXP_QUANT_SCALE;
static constexpr float kInvSqrtHD = 0.125f;

static constexpr int kRowBytes = 2 * kHD + 8;
static constexpr int kVOff = kHD;
static constexpr int kKScaleOff = 2 * kHD;
static constexpr int kVScaleOff = 2 * kHD + 4;

// State: a 64 B header (int32 count of folded blocks), then per Q head a
// 64 B-aligned record [max f32 | sum f32 | pad | sv f32[kHD]].
static constexpr int kStateHeader = 64;
static constexpr int kHeadStride = 64 + 4 * kHD;
static constexpr int kStateBytes = kStateHeader + kREP * kHeadStride;

static_assert(kHD == 64, "kInvSqrtHD is hardcoded for head_dim=64");
static_assert(kTB % kLanes == 0, "block rows must be a multiple of 16");

static inline float bits_to_float(int32_t bits) {
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static inline int32_t float_to_bits(float f) {
  int32_t bits;
  memcpy(&bits, &f, 4);
  return bits;
}

// See llama_flowkv.cc for why Peano's `/` needs this.
static inline float sw_recip(float a) {
  int32_t bits = (int32_t)0x7EF477D5 - float_to_bits(a);
  float x = bits_to_float(bits);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

// e^x for x <= 0: 2^k * 2^f with k = trunc(x * log2(e)) and f in (-1, 0],
// 2^f by its degree-9 Taylor series (relative error below 1e-8). Below
// 2^-125 it returns 0.
static inline float sw_exp_nonpos(float x) {
  float t = x * 1.442695022f;
  if (t < -125.0f)
    return 0.0f;
  int32_t k = (int32_t)t;
  float f = t - (float)k;
  float p = 1.017808628e-07f;
  p = p * f + 1.321548666e-06f;
  p = p * f + 1.525273365e-05f;
  p = p * f + 1.540352969e-04f;
  p = p * f + 1.333355787e-03f;
  p = p * f + 9.618128650e-03f;
  p = p * f + 5.550410971e-02f;
  p = p * f + 2.402265072e-01f;
  p = p * f + 6.931471825e-01f;
  p = p * f + 1.0f;
  return bits_to_float(float_to_bits(p) + (k << 23));
}

static inline float *head_record(int8_t *state, int h) {
  return reinterpret_cast<float *>(state + kStateHeader + h * kHeadStride);
}

// Folds (m_b, s_b, sv_b) into (m, s, sv): with M = max(m, m_b),
// a = s e^(m-M), b = s_b e^(m_b-M), sv = sv a/(a+b) + sv_b b/(a+b).
static inline void fold(float *rec, float m_b, float s_b,
                        const float *restrict sv_b) {
  float m = rec[0];
  float s = rec[1];
  float big = ::aie::max(m, m_b);
  float a = s * sw_exp_nonpos(m - big);
  float b = s_b * sw_exp_nonpos(m_b - big);
  float den = a + b;
  float inv = sw_recip(den);
  ::aie::vector<float, kLanes> wa = ::aie::broadcast<float, kLanes>(a * inv);
  ::aie::vector<float, kLanes> wb = ::aie::broadcast<float, kLanes>(b * inv);
  float *sv = rec + 16;
  for (int j = 0; j < kSlices; j++)
    ::aie::store_v(
        sv + j * kLanes,
        ::aie::add(vec_mul_rne(::aie::load_v<kLanes>(sv + j * kLanes), wa),
                   vec_mul_rne(::aie::load_v<kLanes>(sv_b + j * kLanes), wb)));
  rec[0] = big;
  rec[1] = den;
}

alignas(64) static const int32_t kLaneIndex[kLanes] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

extern "C" {

void llama_attn_state_init(int8_t *restrict state) {
  memset(state, 0, kStateBytes);
}

// Attends q_chunk's kREP heads (kREP*kHD int8, then per head [q_scale f32 |
// unused f32]) over the first n_rows rows of block and folds the result
// into state.
void llama_attn_block(int8_t *restrict q_chunk, int32_t *restrict block_i32,
                      int8_t *restrict state, int32_t n_rows) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const int8_t *block = reinterpret_cast<const int8_t *>(block_i32);
  const int8_t *tail = q_chunk + kREP * kHD;
  int32_t count;
  memcpy(&count, state, 4);

  alignas(64) int32_t dots[kTB];
  alignas(64) int32_t kslot[kTB];
  alignas(64) int32_t vslot[kTB];
  alignas(64) int32_t scores[kTB];
  alignas(64) int32_t expq[kTB];
  alignas(64) int32_t pv[kTB];
  alignas(64) float sv_b[kHD];

  for (int r = 0; r < n_rows; r++) {
    kslot[r] = block_i32[(r * kRowBytes + kKScaleOff) / 4];
    vslot[r] = block_i32[(r * kRowBytes + kVScaleOff) / 4];
  }
  const int groups = (n_rows + kLanes - 1) / kLanes;

  for (int h = 0; h < kREP; h++) {
    float q_scale;
    memcpy(&q_scale, tail + h * 8, 4);
    ::aie::vector<int8, kHD> qv = ::aie::load_v<kHD>(q_chunk + h * kHD);
    for (int r = 0; r < n_rows; r++) {
      ::aie::accum<acc32, kHD> acc;
      acc.from_vector(::aie::zeros<int32, kHD>());
      acc = ::aie::mac(acc, qv,
                       ::aie::load_unaligned_v<kHD>(block + r * kRowBytes));
      dots[r] = ::aie::reduce_add(acc.template to_vector<int32>());
    }

    // score = dot * ((q_scale * k_slot) * 1/sqrt(hd)); rows past n_rows
    // read as -1e30 so they never win the max.
    ::aie::vector<float, kLanes> q_v = ::aie::broadcast<float, kLanes>(q_scale);
    ::aie::vector<float, kLanes> isq_v =
        ::aie::broadcast<float, kLanes>(kInvSqrtHD);
    ::aie::vector<float, kLanes> floor_v =
        ::aie::broadcast<float, kLanes>(-1e30f);
    ::aie::vector<float, kLanes> max_v = floor_v;
    for (int g = 0; g < groups; g++) {
      ::aie::vector<float, kLanes> qk =
          vec_mul_rne(vec_mul_rne(q_v, ::aie::load_v<kLanes>(kslot + g * kLanes)
                                           .template cast_to<float>()),
                      isq_v);
      ::aie::vector<float, kLanes> s = vec_mul_rne(
          ::aie::to_float<float>(::aie::load_v<kLanes>(dots + g * kLanes), 0),
          qk);
      s = ::aie::select(
          floor_v, s,
          ::aie::lt(::aie::load_v<kLanes>(kLaneIndex),
                    ::aie::broadcast<int32, kLanes>(n_rows - g * kLanes)));
      ::aie::store_v(scores + g * kLanes, s.template cast_to<int32>());
      max_v = ::aie::max(max_v, s);
    }
    float m_b = ::aie::reduce_max(max_v);

    // q = round((score - max) / exp_quant_scale) in [-128, 0], e = LUT[q].
    ::aie::vector<float, kLanes> m_v = ::aie::broadcast<float, kLanes>(m_b);
    ::aie::vector<float, kLanes> iq_v =
        ::aie::broadcast<float, kLanes>(kInvExpQuantScale);
    for (int g = 0; g < groups; g++) {
      ::aie::vector<int8, kLanes> q8 = vec_round_to_i8(
          vec_mul_rne(::aie::sub(::aie::load_v<kLanes>(scores + g * kLanes)
                                     .template cast_to<float>(),
                                 m_v),
                      iq_v));
      for (int i = 0; i < kLanes; i++)
        expq[g * kLanes + i] = (int32_t)llama_exp_lut_raw[(int)q8[i] + 128];
    }
    float sum = 0.0f;
    for (int r = 0; r < n_rows; r++)
      sum += bits_to_float(expq[r]);
    ::aie::vector<float, kLanes> inv_v =
        ::aie::broadcast<float, kLanes>(sw_recip(sum));

    // pv = (e * inv_sum) * v_slot, per row.
    for (int g = 0; g < groups; g++)
      ::aie::store_v(
          pv + g * kLanes,
          vec_mul_rne(vec_mul_rne(::aie::load_v<kLanes>(expq + g * kLanes)
                                      .template cast_to<float>(),
                                  inv_v),
                      ::aie::load_v<kLanes>(vslot + g * kLanes)
                          .template cast_to<float>())
              .template cast_to<int32>());

    // sv = sum over rows of pv[r] * v[r], in row order. Four named
    // accumulators: Peano AIE2P corrupts an fp32 vector array kept live
    // across a loop.
    ::aie::vector<float, kLanes> acc0 = ::aie::zeros<float, kLanes>();
    ::aie::vector<float, kLanes> acc1 = ::aie::zeros<float, kLanes>();
    ::aie::vector<float, kLanes> acc2 = ::aie::zeros<float, kLanes>();
    ::aie::vector<float, kLanes> acc3 = ::aie::zeros<float, kLanes>();
    for (int r = 0; r < n_rows; r++) {
      ::aie::vector<int8, kHD> vv =
          ::aie::load_unaligned_v<kHD>(block + r * kRowBytes + kVOff);
      ::aie::vector<float, kLanes> pv_v =
          ::aie::broadcast<float, kLanes>(bits_to_float(pv[r]));
      acc0 = ::aie::add(acc0, vec_mul_rne(::aie::to_float<float>(
                                              vv.template extract<kLanes>(0)),
                                          pv_v));
      acc1 = ::aie::add(acc1, vec_mul_rne(::aie::to_float<float>(
                                              vv.template extract<kLanes>(1)),
                                          pv_v));
      acc2 = ::aie::add(acc2, vec_mul_rne(::aie::to_float<float>(
                                              vv.template extract<kLanes>(2)),
                                          pv_v));
      acc3 = ::aie::add(acc3, vec_mul_rne(::aie::to_float<float>(
                                              vv.template extract<kLanes>(3)),
                                          pv_v));
    }

    float *rec = head_record(state, h);
    float *dst = count == 0 ? rec + 16 : sv_b;
    ::aie::store_v(dst, acc0);
    ::aie::store_v(dst + kLanes, acc1);
    ::aie::store_v(dst + 2 * kLanes, acc2);
    ::aie::store_v(dst + 3 * kLanes, acc3);
    if (count == 0) {
      rec[0] = m_b;
      rec[1] = sum;
    } else {
      fold(rec, m_b, sum, sv_b);
    }
  }

  count += 1;
  memcpy(state, &count, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// The growing decode's chunk carries the token's own K/V row after the
// q_chunk (llama_qkv_combine_rows): llama_attn_block over a block whose row
// n_rows - 1 is first set to that row, which is not in the cache yet.
static constexpr int kQChunkBytes = kREP * kHD + kREP * 8;

void llama_attn_block_own(int8_t *restrict q_chunk, int32_t *restrict block_i32,
                          int8_t *restrict state, int32_t n_rows) {
  memcpy(block_i32 + (n_rows - 1) * (kRowBytes / 4), q_chunk + kQChunkBytes,
         kRowBytes);
  llama_attn_block(q_chunk, block_i32, state, n_rows);
}

// The token's own row, for the cache write-back.
void llama_attn_own_row(int8_t *restrict q_chunk, int32_t *restrict row_i32) {
  memcpy(row_i32, q_chunk + kQChunkBytes, kRowBytes);
}

// Folds n_parts states (each kStateBytes, in core order; empty ones are
// skipped) and requantizes each Q head's sv with its own scale:
// out_chunk = kREP*kHD int8, then kREP f32 sv_out_scales.
void llama_attn_merge_selfcal(int8_t *restrict parts, int8_t *restrict out,
                              int32_t n_parts) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  int first = -1;
  for (int c = 0; c < n_parts && first < 0; c++) {
    int32_t count;
    memcpy(&count, parts + c * kStateBytes, 4);
    if (count > 0)
      first = c;
  }
  int8_t *acc_state = parts + first * kStateBytes;
  for (int c = first + 1; c < n_parts; c++) {
    int8_t *part = parts + c * kStateBytes;
    int32_t count;
    memcpy(&count, part, 4);
    if (count == 0)
      continue;
    for (int h = 0; h < kREP; h++) {
      float *src = head_record(part, h);
      fold(head_record(acc_state, h), src[0], src[1], src + 16);
    }
  }

  for (int h = 0; h < kREP; h++) {
    const float *sv = head_record(acc_state, h) + 16;
    ::aie::vector<int32, kLanes> mag = ::aie::zeros<int32, kLanes>();
    for (int j = 0; j < kSlices; j++)
      mag =
          ::aie::max(mag, ::aie::bit_and(int32(0x7fffffff),
                                         ::aie::load_v<kLanes>(sv + j * kLanes)
                                             .template cast_to<int32>()));
    float absmax = bits_to_float(::aie::reduce_max(mag));
    if (absmax < 1e-12f)
      absmax = 1e-12f;
    float scale = absmax * (1.0f / 127.0f);
    ::aie::vector<float, kLanes> inv_v =
        ::aie::broadcast<float, kLanes>(sw_recip(scale));
    for (int j = 0; j < kSlices; j++)
      ::aie::store_v(out + h * kHD + j * kLanes,
                     vec_round_to_i8(vec_mul_rne(
                         ::aie::load_v<kLanes>(sv + j * kLanes), inv_v)));
    memcpy(out + kREP * kHD + h * 4, &scale, 4);
  }

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// llama_attn_merge_selfcal over two cores' states, the first core's first.
void llama_attn_merge_pair_selfcal(int8_t *restrict state0,
                                   int8_t *restrict state1,
                                   int8_t *restrict out) {
  alignas(64) int8_t parts[2 * kStateBytes];
  memcpy(parts, state0, kStateBytes);
  memcpy(parts + kStateBytes, state1, kStateBytes);
  llama_attn_merge_selfcal(parts, out, 2);
}

} // extern "C"
