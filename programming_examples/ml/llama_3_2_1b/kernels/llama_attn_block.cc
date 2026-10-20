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

alignas(64) static const int32_t kLaneIndex[kLanes] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

static inline float *head_record(int8_t *state, int h) {
  return reinterpret_cast<float *>(state + kStateHeader + h * kHeadStride);
}

// sw_exp_nonpos on kLanes lanes. Its products are vec_mul_rne, which differ
// from the scalar soft-float ones only on denormals that the next vadd.f
// absorbs. float2fix gives floor(t) or ceil(t); trunc(t) for t <= 0 is the
// ceiling, so a negative n below t (compared as sign-magnitude bits) is
// bumped by one.
static inline ::aie::vector<float, kLanes>
vec_sw_exp_nonpos(::aie::vector<float, kLanes> x) {
  ::aie::vector<float, kLanes> t =
      vec_mul_rne(x, ::aie::broadcast<float, kLanes>(1.442695022f));
  ::aie::vector<int32, kLanes> tb = t.template cast_to<int32>();
  ::aie::vector<int32, kLanes> n = ::aie::to_fixed<int32>(t, 0);
  ::aie::vector<int32, kLanes> nb =
      ::aie::to_float<float>(n, 0).template cast_to<int32>();
  ::aie::vector<int32, kLanes> k = ::aie::select(
      n, ::aie::add(n, int32(1)), ::aie::gt(nb, tb) & ::aie::lt(nb, int32(0)));
  ::aie::vector<float, kLanes> f = ::aie::sub(t, ::aie::to_float<float>(k, 0));
  static constexpr float kC[] = {
      1.321548666e-06f, 1.525273365e-05f, 1.540352969e-04f,
      1.333355787e-03f, 9.618128650e-03f, 5.550410971e-02f,
      2.402265072e-01f, 6.931471825e-01f, 1.0f};
  ::aie::vector<float, kLanes> p =
      ::aie::broadcast<float, kLanes>(1.017808628e-07f);
  for (int i = 0; i < 9; i++)
    p = ::aie::add(vec_mul_rne(p, f), ::aie::broadcast<float, kLanes>(kC[i]));
  return ::aie::select(
             ::aie::add(p.template cast_to<int32>(), ::aie::upshift(k, 23)),
             ::aie::zeros<int32, kLanes>(),
             ::aie::lt(tb, int32(0)) &
                 ::aie::gt(tb, __builtin_bit_cast(int32_t, -125.0f)))
      .template cast_to<float>();
}

// Folds every head's block max and sum (m_b[h], s_b[h]) into its record:
// with M = max(m, m_b), a = s e^(m-M) and b = s_b e^(m_b-M), the record gets
// (M, a + b) and w the weights a/(a+b) (w[h]) and b/(a+b) (w[kREP + h]) of
// the old sv and the block's. Lanes h and kREP + h carry each head's two
// sides through one exp. A weight becomes 0 where the scalar product is a
// denormal, which vec_mul_rne treats as 0 anyway.
static void fold_weights(int8_t *state, const float *m_b, const float *s_b,
                         float *w) {
  static_assert(2 * kREP <= kLanes, "both sides of every head in one vector");
  alignas(64) float x[2 * kLanes] = {};
  for (int h = 0; h < kREP; h++) {
    x[h] = head_record(state, h)[0];
    x[kREP + h] = m_b[h];
    x[kLanes + h] = head_record(state, h)[1];
    x[kLanes + kREP + h] = s_b[h];
  }
  auto low = ::aie::lt(::aie::load_v<kLanes>(kLaneIndex), int32(kREP));
  ::aie::vector<float, kLanes> m = ::aie::load_v<kLanes>(x);
  ::aie::vector<float, kLanes> big =
      ::aie::max(m, ::aie::select(::aie::shuffle_up(m, kREP),
                                  ::aie::shuffle_down(m, kREP), low));
  ::aie::vector<float, kLanes> ab = vec_mul_rne(
      ::aie::load_v<kLanes>(x + kLanes), vec_sw_exp_nonpos(::aie::sub(m, big)));
  ::aie::vector<float, kLanes> den =
      ::aie::add(ab, ::aie::shuffle_down(ab, kREP));
  ::aie::vector<float, kLanes> inv = vec_sw_recip(den);
  ::aie::store_v(w, vec_mul_rne(ab, ::aie::select(::aie::shuffle_up(inv, kREP),
                                                  inv, low)));
  ::aie::store_v(x, big);
  ::aie::store_v(x + kLanes, den);
  for (int h = 0; h < kREP; h++) {
    head_record(state, h)[0] = x[h];
    head_record(state, h)[1] = x[kLanes + h];
  }
}

// sv = sv * wa + sv_b * wb over one head's kHD lanes. Out of line, like the
// rolled loops here, to fit the attention core's 16 KB program memory.
__attribute__((noinline)) static void fold_sv(float *sv, const float *sv_b,
                                              float wa, float wb) {
  ::aie::vector<float, kLanes> wa_v = ::aie::broadcast<float, kLanes>(wa);
  ::aie::vector<float, kLanes> wb_v = ::aie::broadcast<float, kLanes>(wb);
#pragma clang loop unroll(disable)
  for (int j = 0; j < kSlices; j++)
    ::aie::store_v(
        sv + j * kLanes,
        ::aie::add(
            vec_mul_rne(::aie::load_v<kLanes>(sv + j * kLanes), wa_v),
            vec_mul_rne(::aie::load_v<kLanes>(sv_b + j * kLanes), wb_v)));
}

// Folds part's heads into acc_state's.
static void fold_part(int8_t *acc_state, int8_t *part) {
  alignas(64) float m_b[kREP];
  alignas(64) float s_b[kREP];
  for (int h = 0; h < kREP; h++) {
    m_b[h] = head_record(part, h)[0];
    s_b[h] = head_record(part, h)[1];
  }
  alignas(64) float w[kLanes];
  fold_weights(acc_state, m_b, s_b, w);
  for (int h = 0; h < kREP; h++)
    fold_sv(head_record(acc_state, h) + 16, head_record(part, h) + 16, w[h],
            w[kREP + h]);
}

// Requantizes each Q head's sv in state with its own scale: out = kREP*kHD
// int8, then kREP f32 sv_out_scales.
static void requant_selfcal(int8_t *state, int8_t *out) {
  // Each head's scale max(absmax, 1e-12) * (1/127) and its reciprocal at
  // once; max(absmax, 1e-12f) is an integer max of the non-negative bits.
  alignas(64) int32_t absmax[kLanes] = {};
#pragma clang loop unroll(disable)
  for (int i = 0; i < kREP * kSlices; i++) {
    const float *sv =
        head_record(state, i / kSlices) + 16 + i % kSlices * kLanes;
    int32_t mag = ::aie::reduce_max(
        ::aie::bit_and(int32(0x7fffffff),
                       ::aie::load_v<kLanes>(sv).template cast_to<int32>()));
    if (mag > absmax[i / kSlices])
      absmax[i / kSlices] = mag;
  }
  ::aie::vector<float, kLanes> scale =
      vec_mul_rne(::aie::max(::aie::load_v<kLanes>(absmax),
                             __builtin_bit_cast(int32_t, 1e-12f))
                      .template cast_to<float>(),
                  ::aie::broadcast<float, kLanes>(1.0f / 127.0f));
  alignas(64) float scale_a[kLanes];
  alignas(64) float inv_a[kLanes];
  ::aie::store_v(scale_a, scale);
  ::aie::store_v(inv_a, vec_sw_recip(scale));
  memcpy(out + kREP * kHD, scale_a, kREP * 4);
#pragma clang loop unroll(disable)
  for (int i = 0; i < kREP * kSlices; i++)
    ::aie::store_v(out + i * kLanes,
                   vec_round_to_i8(vec_mul_rne(
                       ::aie::load_v<kLanes>(head_record(state, i / kSlices) +
                                             16 + i % kSlices * kLanes),
                       ::aie::broadcast<float, kLanes>(inv_a[i / kSlices]))));
}

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
  int32_t &count = *reinterpret_cast<int32_t *>(state);

  alignas(64) int32_t dots[kREP * kTB];
  alignas(64) int32_t kslot[kTB];
  alignas(64) int32_t vslot[kTB];
  alignas(64) int32_t scores[kTB];
  alignas(64) int32_t expq[kREP * kTB];
  alignas(64) int32_t pv[kTB];
  alignas(64) uint16_t pv_mh[kTB];
  alignas(64) uint16_t pv_ml[kTB];
  alignas(64) int32_t pv_e[kTB];
  alignas(64) float sv_b[kHD];

  // Rows past n_rows (still inside the kTB-row block) give scores that the
  // -1e30 select drops and e = 0, so pv = 0 whatever their v_scale.
  const int groups = (n_rows + kLanes - 1) / kLanes;
#pragma clang loop min_iteration_count(16)
  for (int r = 0; r < groups * kLanes; r++) {
    kslot[r] = block_i32[(r * kRowBytes + kKScaleOff) / 4];
    vslot[r] = block_i32[(r * kRowBytes + kVScaleOff) / 4];
  }

  // dots[h][r] = q_h . k_r for all heads and eight rows per step, on
  // mmul<4, 8, 8>: A is q's 4x8 tile of columns 8j..8j+7, B the eight rows'
  // same columns transposed, built from the rows by three zip stages.
  alignas(64) int8_t q_tiles[kREP * kHD];
  {
    auto [a0, a1] = ::aie::interleave_zip(::aie::load_v<kHD>(q_chunk),
                                          ::aie::load_v<kHD>(q_chunk + kHD), 8);
    auto [b0, b1] =
        ::aie::interleave_zip(::aie::load_v<kHD>(q_chunk + 2 * kHD),
                              ::aie::load_v<kHD>(q_chunk + 3 * kHD), 8);
    auto [c0, c1] = ::aie::interleave_zip(a0, b0, 16);
    auto [d0, d1] = ::aie::interleave_zip(a1, b1, 16);
    ::aie::store_v(q_tiles, c0);
    ::aie::store_v(q_tiles + kHD, c1);
    ::aie::store_v(q_tiles + 2 * kHD, d0);
    ::aie::store_v(q_tiles + 3 * kHD, d1);
  }
  for (int g = 0; g < kTB / 8; g++) {
    const int8_t *rows = block + 8 * g * kRowBytes;
    auto [a0, a1] = ::aie::interleave_zip(
        ::aie::load_unaligned_v<kHD>(rows),
        ::aie::load_unaligned_v<kHD>(rows + kRowBytes), 1);
    auto [b0, b1] = ::aie::interleave_zip(
        ::aie::load_unaligned_v<kHD>(rows + 2 * kRowBytes),
        ::aie::load_unaligned_v<kHD>(rows + 3 * kRowBytes), 1);
    auto [c0, c1] = ::aie::interleave_zip(
        ::aie::load_unaligned_v<kHD>(rows + 4 * kRowBytes),
        ::aie::load_unaligned_v<kHD>(rows + 5 * kRowBytes), 1);
    auto [d0, d1] = ::aie::interleave_zip(
        ::aie::load_unaligned_v<kHD>(rows + 6 * kRowBytes),
        ::aie::load_unaligned_v<kHD>(rows + 7 * kRowBytes), 1);
    auto [e0, e1] = ::aie::interleave_zip(a0, b0, 2);
    auto [e2, e3] = ::aie::interleave_zip(a1, b1, 2);
    auto [f0, f1] = ::aie::interleave_zip(c0, d0, 2);
    auto [f2, f3] = ::aie::interleave_zip(c1, d1, 2);
    auto [t0, t1] = ::aie::interleave_zip(e0, f0, 4);
    auto [t2, t3] = ::aie::interleave_zip(e1, f1, 4);
    auto [t4, t5] = ::aie::interleave_zip(e2, f2, 4);
    auto [t6, t7] = ::aie::interleave_zip(e3, f3, 4);
    using MMUL = ::aie::mmul<4, 8, 8, int8, int8, acc32>;
    MMUL c;
    c.mul(::aie::load_v<32>(q_tiles), t0);
    c.mac(::aie::load_v<32>(q_tiles + 32), t1);
    c.mac(::aie::load_v<32>(q_tiles + 64), t2);
    c.mac(::aie::load_v<32>(q_tiles + 96), t3);
    c.mac(::aie::load_v<32>(q_tiles + 128), t4);
    c.mac(::aie::load_v<32>(q_tiles + 160), t5);
    c.mac(::aie::load_v<32>(q_tiles + 192), t6);
    c.mac(::aie::load_v<32>(q_tiles + 224), t7);
    ::aie::vector<int32, 32> d = c.template to_vector<int32>();
    for (int h = 0; h < kREP; h++)
      ::aie::store_v(dots + h * kTB + 8 * g, d.template extract<8>(h));
  }

  // Per head: score = dot * ((q_scale * k_slot) * 1/sqrt(hd)), rows past
  // n_rows read as -1e30 so they never win the max; then the LUT index
  // round((score - max) / exp_quant_scale) in [-128, 0].
  alignas(64) float lanes[2 * kLanes] = {};
  alignas(64) int8_t qidx[kREP * kTB];
  ::aie::vector<float, kLanes> isq_v =
      ::aie::broadcast<float, kLanes>(kInvSqrtHD);
  ::aie::vector<float, kLanes> floor_v =
      ::aie::broadcast<float, kLanes>(-1e30f);
  ::aie::vector<float, kLanes> iq_v =
      ::aie::broadcast<float, kLanes>(kInvExpQuantScale);
  for (int h = 0; h < kREP; h++) {
    const float q_scale = reinterpret_cast<const float *>(tail)[2 * h];
    ::aie::vector<float, kLanes> q_v = ::aie::broadcast<float, kLanes>(q_scale);
    ::aie::vector<float, kLanes> max_v = floor_v;
    for (int g = 0; g < groups; g++) {
      ::aie::vector<float, kLanes> qk =
          vec_mul_rne(vec_mul_rne(q_v, ::aie::load_v<kLanes>(kslot + g * kLanes)
                                           .template cast_to<float>()),
                      isq_v);
      ::aie::vector<float, kLanes> s = vec_mul_rne(
          ::aie::to_float<float>(
              ::aie::load_v<kLanes>(dots + h * kTB + g * kLanes), 0),
          qk);
      s = ::aie::select(
          floor_v, s,
          ::aie::lt(::aie::load_v<kLanes>(kLaneIndex),
                    ::aie::broadcast<int32, kLanes>(n_rows - g * kLanes)));
      ::aie::store_v(scores + g * kLanes, s.template cast_to<int32>());
      max_v = ::aie::max(max_v, s);
    }
    float m_b = ::aie::reduce_max(max_v);
    lanes[h] = m_b;
    ::aie::vector<float, kLanes> m_v = ::aie::broadcast<float, kLanes>(m_b);
    for (int g = 0; g < groups; g++)
      ::aie::store_v(qidx + h * kTB + g * kLanes,
                     vec_round_to_i8(vec_mul_rne(
                         ::aie::sub(::aie::load_v<kLanes>(scores + g * kLanes)
                                        .template cast_to<float>(),
                                    m_v),
                         iq_v)));
  }

  // e = LUT[q] (rows past n_rows as 0); the sums run in row order with a
  // head per lane, on 16-row groups transposed by two zip stages.
  for (int h = 0; h < kREP; h++) {
    const int8_t *q = qidx + h * kTB;
    int32_t *e = expq + h * kTB;
#pragma clang loop min_iteration_count(4)
    for (int r = 0; r < groups * kLanes; r += 4) {
      e[r] = (int32_t)llama_exp_lut_raw[(int)q[r] + 128];
      e[r + 1] = (int32_t)llama_exp_lut_raw[(int)q[r + 1] + 128];
      e[r + 2] = (int32_t)llama_exp_lut_raw[(int)q[r + 2] + 128];
      e[r + 3] = (int32_t)llama_exp_lut_raw[(int)q[r + 3] + 128];
    }
    for (int r = n_rows; r < groups * kLanes; r++)
      expq[h * kTB + r] = 0;
  }
  static_assert(kREP == 4, "the sum transposes four heads");
  ::aie::vector<float, kLanes> sum_v = ::aie::zeros<float, kLanes>();
  for (int g = 0; g < groups; g++) {
    auto [a0, a1] = ::aie::interleave_zip(
        ::aie::load_v<kLanes>(expq + g * kLanes),
        ::aie::load_v<kLanes>(expq + kTB + g * kLanes), 1);
    auto [b0, b1] = ::aie::interleave_zip(
        ::aie::load_v<kLanes>(expq + 2 * kTB + g * kLanes),
        ::aie::load_v<kLanes>(expq + 3 * kTB + g * kLanes), 1);
    auto [t0, t1] = ::aie::interleave_zip(a0, b0, 2);
    auto [t2, t3] = ::aie::interleave_zip(a1, b1, 2);
    for (::aie::vector<int32, kLanes> t : {t0, t1, t2, t3}) {
      ::aie::vector<float, kLanes> e = t.template cast_to<float>();
      sum_v = ::aie::add(sum_v, e);
      sum_v = ::aie::add(sum_v, ::aie::shuffle_down(e, 4));
      sum_v = ::aie::add(sum_v, ::aie::shuffle_down(e, 8));
      sum_v = ::aie::add(sum_v, ::aie::shuffle_down(e, 12));
    }
  }
  alignas(64) float inv_a[kLanes];
  ::aie::store_v(inv_a, vec_sw_recip(sum_v));
  ::aie::store_v(lanes + kLanes, sum_v);
  alignas(64) float w[kLanes];
  if (count == 0) {
    for (int h = 0; h < kREP; h++) {
      head_record(state, h)[0] = lanes[h];
      head_record(state, h)[1] = lanes[kLanes + h];
    }
  } else {
    fold_weights(state, lanes, lanes + kLanes, w);
  }

  for (int h = 0; h < kREP; h++) {
    ::aie::vector<float, kLanes> inv_v =
        ::aie::broadcast<float, kLanes>(inv_a[h]);

    // pv = (e * inv_sum) * v_slot, per row, with its significand split into
    // 16 + 8 bits for the products below (0 past n_rows) and the to_float
    // shift of its exponent.
    ::aie::vector<int32, kLanes> pv_min =
        ::aie::broadcast<int32, kLanes>(0x7fffffff);
    ::aie::vector<int32, kLanes> pv_max = ::aie::zeros<int32, kLanes>();
    for (int g = 0; g < groups; g++) {
      ::aie::vector<int32, kLanes> b =
          vec_mul_rne(
              vec_mul_rne(::aie::load_v<kLanes>(expq + h * kTB + g * kLanes)
                              .template cast_to<float>(),
                          inv_v),
              ::aie::load_v<kLanes>(vslot + g * kLanes)
                  .template cast_to<float>())
              .template cast_to<int32>();
      ::aie::store_v(pv + g * kLanes, b);
      auto is_live =
          ::aie::lt(::aie::load_v<kLanes>(kLaneIndex),
                    ::aie::broadcast<int32, kLanes>(n_rows - g * kLanes));
      ::aie::vector<int32, kLanes> live = ::aie::select(
          ::aie::broadcast<int32, kLanes>(0x3f800000), b, is_live);
      pv_min = ::aie::min(pv_min, live);
      pv_max = ::aie::max(pv_max, live);
      ::aie::vector<int32, kLanes> mant = ::aie::select(
          ::aie::zeros<int32, kLanes>(),
          ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), b)),
          is_live);
      ::aie::accum<acc32, kLanes> t;
      // downshift rounds by the rounding mode, so only whole bits go in.
      t.from_vector(::aie::downshift(::aie::bit_and(int32(0xffff00), mant), 8));
      ::aie::store_v(pv_mh + g * kLanes, t.template to_vector<uint16>(0));
      t.from_vector(::aie::bit_and(int32(0xff), mant));
      ::aie::store_v(pv_ml + g * kLanes, t.template to_vector<uint16>(0));
      ::aie::store_v(
          pv_e + g * kLanes,
          ::aie::downshift(::aie::bit_and(int32(0x7f800000), live), 23));
    }
    // |v| * pv rounded once: n = |v| * significand is exact in acc32 as
    // (|v| << 8) * (m >> 8) + |v| * (m & 0xff), and to_float(n, 150 - e)
    // rounds it to nearest-even (checked on NPU2 for every |v| and every
    // significand). Needs every pv positive with its exponent field in
    // [24, 238], so that to_float's halves stay normal.
    const bool fast = ::aie::reduce_min(pv_min) >= (24 << 23) &&
                      ::aie::reduce_max(pv_max) < (239 << 23);

    // sv = sum over rows of pv[r] * v[r], in row order, into the record or,
    // to be folded in, into sv_b.
    float *sv = head_record(state, h) + 16;
    float *dst = count == 0 ? sv : sv_b;
    if (fast) {
      ::aie::vector<float, 32> sv_lo = ::aie::zeros<float, 32>();
      ::aie::vector<float, 32> sv_hi = ::aie::zeros<float, 32>();
      // Rows past n_rows have a zero significand, so they add a zero.
#pragma clang loop min_iteration_count(16)
      for (int r = 0; r < groups * kLanes; r++) {
        ::aie::vector<int16, kHD> v16 =
            ::aie::load_unaligned_v<kHD>(block + r * kRowBytes + kVOff)
                .unpack();
        ::aie::vector<uint16, kHD> av =
            ::aie::abs(v16).template cast_to<uint16>();
        ::aie::vector<uint16, kHD> av8 = ::aie::upshift(av, 8);
        const uint16_t mh = pv_mh[r], ml = pv_ml[r];
        const int sh = 150 - pv_e[r];
        ::aie::accum<acc32, 32> n0 =
            ::aie::mul(av8.template extract<32>(0), mh);
        ::aie::accum<acc32, 32> n1 =
            ::aie::mul(av8.template extract<32>(1), mh);
        n0 = ::aie::mac(n0, av.template extract<32>(0), ml);
        n1 = ::aie::mac(n1, av.template extract<32>(1), ml);
        ::aie::vector<float, 32> f0 =
            ::aie::to_float<float>(n0.template to_vector<int32>(0), sh);
        ::aie::vector<float, 32> f1 =
            ::aie::to_float<float>(n1.template to_vector<int32>(0), sh);
        sv_lo = ::aie::add(
            sv_lo,
            ::aie::select(f0, ::aie::neg(f0),
                          ::aie::lt(v16.template extract<32>(0), int16(0))));
        sv_hi = ::aie::add(
            sv_hi,
            ::aie::select(f1, ::aie::neg(f1),
                          ::aie::lt(v16.template extract<32>(1), int16(0))));
      }
      ::aie::store_v(dst, sv_lo);
      ::aie::store_v(dst + 32, sv_hi);
    } else {
      for (int j = 0; j < kSlices; j++)
        ::aie::store_v(dst + j * kLanes, ::aie::zeros<float, kLanes>());
      for (int r = 0; r < n_rows; r++) {
        ::aie::vector<float, kLanes> pv_v =
            ::aie::broadcast<float, kLanes>(__builtin_bit_cast(float, pv[r]));
        const int8_t *v = block + r * kRowBytes + kVOff;
#pragma clang loop unroll(disable)
        for (int j = 0; j < kSlices; j++)
          ::aie::store_v(
              dst + j * kLanes,
              ::aie::add(::aie::load_v<kLanes>(dst + j * kLanes),
                         vec_mul_rne(::aie::to_float<float>(
                                         ::aie::load_unaligned_v<kLanes>(
                                             v + j * kLanes)),
                                     pv_v)));
      }
    }
    if (count > 0)
      fold_sv(sv, sv_b, w[h], w[kREP + h]);
  }

  count += 1;

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// A batch of decode tokens' queries: each token's chunk is its q_chunk then
// its own K/V row (llama_comb_rope_rows), and query r's chunk and state sit at
// r * kChunkStride and r * kStateBytes of the core's buffers (llama_attn_block
// loads q aligned).
static constexpr int kQChunkBytes = kREP * kHD + kREP * 8;
static constexpr int kChunkBytes = kQChunkBytes + kRowBytes;
static constexpr int kChunkStride = (kChunkBytes + 63) / 64 * 64;

// Query r of a batch: its chunk into the core's chunks, its state fresh.
void llama_attn_take_q(int8_t *restrict chunk, int8_t *restrict chunks,
                       int8_t *restrict states, int32_t r) {
  memcpy(chunks + r * kChunkStride, chunk, kChunkBytes);
  llama_attn_state_init(states + r * kStateBytes);
}

// llama_attn_block for query r over a block: all kTB rows, or, when row0 >= 0,
// the batch's own block, in which query r attends rows up to row0 + r, its
// own K/V row (not in the cache yet) first set there; the earlier queries,
// run first, set theirs.
void llama_attn_block_q(int8_t *restrict chunks, int32_t *restrict block_i32,
                        int8_t *restrict states, int32_t row0, int32_t r) {
  int8_t *chunk = chunks + r * kChunkStride;
  if (row0 >= 0)
    memcpy(block_i32 + (row0 + r) * (kRowBytes / 4), chunk + kQChunkBytes,
           kRowBytes);
  llama_attn_block(chunk, block_i32, states + r * kStateBytes,
                   row0 >= 0 ? row0 + r + 1 : kTB);
}

void llama_attn_put_q(int8_t *restrict states, int8_t *restrict part,
                      int32_t r) {
  memcpy(part, states + r * kStateBytes, kStateBytes);
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
    if (count > 0)
      fold_part(acc_state, part);
  }
  requant_selfcal(acc_state, out);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// llama_attn_merge_selfcal over two cores' states, the first core's first,
// folding in place into the first non-empty one.
void llama_attn_merge_pair_selfcal(int8_t *restrict state0,
                                   int8_t *restrict state1,
                                   int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  int32_t count0, count1;
  memcpy(&count0, state0, 4);
  memcpy(&count1, state1, 4);
  if (count0 > 0 && count1 > 0)
    fold_part(state0, state1);
  requant_selfcal(count0 > 0 ? state0 : state1, out);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// llama_attn_merge_pair_selfcal for query r: the first core's state, then
// the second's at states + r * kStateBytes.
void llama_attn_merge_pair_q(int8_t *restrict state0, int8_t *restrict states,
                             int8_t *restrict out, int32_t r) {
  llama_attn_merge_pair_selfcal(state0, states + r * kStateBytes, out);
}

} // extern "C"
