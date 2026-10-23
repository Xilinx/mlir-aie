//===- llama_attn_block.cc ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Block-wise GQA attention for one KV head's kREP Q heads over a cache of
// kTB-row blocks, a row being [k int8[kHD] | v int8[kHD] | k_scale | v_scale].
// Each block is attended as a whole cache would be, then folded into a
// running (max, sum, normalized sv) state; products round once as the scalar
// `*` does and sums are vadd.f, so llama_oracle.py mirrors it exactly.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

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

static constexpr int kHD = LLAMA_ATTN_HEAD_DIM;
static constexpr int kTB = LLAMA_ATTN_TB;
static constexpr int kREP = LLAMA_ATTN_REP;
static constexpr int kLanes = 16;
static constexpr int kSlices = kHD / kLanes;

static constexpr int kRowBytes = 2 * kHD + 8;
static constexpr int kVOff = kHD;
static constexpr int kKScaleOff = 2 * kHD;
static constexpr int kVScaleOff = 2 * kHD + 4;

// State: a 64 B header (int32 count of folded blocks), then per Q head a
// 64 B-aligned record [max f32 | sum f32 | pad | sv f32[kHD]].
static constexpr int kStateHeader = 64;
static constexpr int kHeadStride = 64 + 4 * kHD;
static constexpr int kStateBytes = kStateHeader + kREP * kHeadStride;

static_assert(kHD == 64, "scores scale by 1/sqrt(hd) = 2^-3");
static_assert(
    kTB % (2 * kLanes) == 0,
    "block rows must be a multiple of 32: exp takes two groups a call");

alignas(64) static const int32_t kLaneIndex[kLanes] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

static inline float *head_record(int8_t *state, int h) {
  return reinterpret_cast<float *>(state + kStateHeader + h * kHeadStride);
}

// recip_rne lane-wise: the same restoring division of the significand, on
// 16 lanes at once.
__attribute__((noinline)) static ::aie::vector<float, kLanes>
vec_recip_rne(::aie::vector<float, kLanes> a) {
  ::aie::vector<int32, kLanes> bits = a.template cast_to<int32>();
  ::aie::vector<int32, kLanes> e =
      ::aie::downshift(::aie::bit_and(int32(0x7f800000), bits), 23);
  ::aie::vector<int32, kLanes> m =
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bits));
  ::aie::vector<int32, kLanes> rem = ::aie::broadcast<int32, kLanes>(1 << 24);
  ::aie::vector<int32, kLanes> q = ::aie::zeros<int32, kLanes>();
  for (int i = 0; i < 24; i++) {
    auto ge = ::aie::ge(rem, m);
    q = ::aie::add(::aie::add(q, q),
                   ::aie::select(::aie::zeros<int32, kLanes>(),
                                 ::aie::broadcast<int32, kLanes>(1), ge));
    rem = ::aie::select(rem, ::aie::sub(rem, m), ge);
    rem = ::aie::add(rem, rem);
  }
  auto exact = ::aie::eq(q, int32(1 << 23)) & ::aie::eq(rem, int32(0));
  auto up =
      ::aie::gt(rem, m) |
      (::aie::eq(rem, m) & ::aie::gt(::aie::bit_and(int32(1), q), int32(0)));
  q = ::aie::select(q, ::aie::add(q, int32(1)), up);
  return ::aie::select(::aie::add(::aie::upshift(::aie::sub(int32(253), e), 23),
                                  ::aie::sub(q, int32(1 << 23))),
                       ::aie::upshift(::aie::sub(int32(254), e), 23), exact)
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
  ::aie::vector<float, kLanes> ab =
      vec_mul_rne(::aie::load_v<kLanes>(x + kLanes),
                  vec_exp_nonpos(::aie::concat(::aie::sub(m, big),
                                               ::aie::zeros<float, kLanes>()))
                      .template extract<kLanes>(0));
  ::aie::vector<float, kLanes> den =
      ::aie::add(ab, ::aie::shuffle_down(ab, kREP));
  // w only reads inv[0, kREP): w[h] = ab[h] * inv[h], and w[kREP + h] =
  // ab[kREP + h] * shuffle_up(inv, kREP)[kREP + h] = ab[kREP + h] * inv[h].
  ::aie::vector<float, kLanes> inv = vec_recip_rne(den);
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

// One head's sv over the first `rows` rows of v (each row's kHD int8 v, at
// kRowBytes strides), in row order, to dst, for a block whose live nonzero
// pv all have exponent fields from 24 to 238: sv_head_low's products without
// its 2^64 offset, its to_float(n, 150 - e) written out in accumulators.
// n = |v| * m is exact in acc64; its high and low 16 bits are each added in
// acc32 to the bits of to_float's magic numbers for that shift, less which in
// accfloat they are exact, and their one vadd.f is to_float's. Per row, m is
// pv's significand and kh and kl the two magic numbers' high 16 bits,
// doubled; v's sign is added to the product's bits, as aie::neg does.
__attribute__((noinline)) static void sv_head(const int8_t *v, int rows,
                                              const int32_t *m,
                                              const uint16_t *kh,
                                              const uint16_t *kl, float *dst) {
  // srs takes n's high half floored and its low half unsaturated; the fp
  // adds round to nearest-even in any mode.
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::floor);
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::none);
  const ::aie::vector<uint16, kHD> c15 = ::aie::broadcast<uint16, kHD>(32768);
  ::aie::accum<accfloat, 32> sv0, sv1;
  sv0.from_vector(::aie::zeros<float, 32>());
  sv1.from_vector(::aie::zeros<float, 32>());
#pragma clang loop min_iteration_count(16)
  for (int r = 0; r < rows; r++) {
    ::aie::vector<int16, kHD> v16 =
        ::aie::load_unaligned_v<kHD>(v + r * kRowBytes).unpack();
    ::aie::vector<uint16, kHD> av = ::aie::abs(v16).template cast_to<uint16>();
    ::aie::accum<acc32, kHD> sg;
    sg.from_vector(::aie::bit_and(int16(-32768), v16), 16);
    // Lanes 0-31 hold the high half's magic, 32-63 the low half's.
    ::aie::accum<acc32, kHD> mag =
        ::aie::mul(c15, ::aie::concat(::aie::broadcast<uint16, 32>(kh[r]),
                                      ::aie::broadcast<uint16, 32>(kl[r])));
    ::aie::accum<acc64, 32> n0 =
        ::aie::mul(av.template extract<32>(0).template cast_to<int16>(), m[r]);
    ::aie::accum<acc64, 32> n1 =
        ::aie::mul(av.template extract<32>(1).template cast_to<int16>(), m[r]);
    ::aie::accum<accfloat, kHD> p0 =
        ::aie::sub(::aie::mac(mag,
                              ::aie::concat(n0.template to_vector<uint16>(16),
                                            n0.template to_vector<uint16>(0)),
                              uint16(1))
                       .template cast_to<accfloat>(),
                   mag.template cast_to<accfloat>());
    ::aie::accum<accfloat, kHD> p1 =
        ::aie::sub(::aie::mac(mag,
                              ::aie::concat(n1.template to_vector<uint16>(16),
                                            n1.template to_vector<uint16>(0)),
                              uint16(1))
                       .template cast_to<accfloat>(),
                   mag.template cast_to<accfloat>());
    ::aie::accum<accfloat, 32> f0 =
        ::aie::add(p0.template extract<32>(1), p0.template extract<32>(0));
    ::aie::accum<accfloat, 32> f1 =
        ::aie::add(p1.template extract<32>(1), p1.template extract<32>(0));
    sv0 = ::aie::add(sv0, ::aie::add(f0.template cast_to<acc32>(),
                                     sg.template extract<32>(0))
                              .template cast_to<accfloat>());
    sv1 = ::aie::add(sv1, ::aie::add(f1.template cast_to<acc32>(),
                                     sg.template extract<32>(1))
                              .template cast_to<accfloat>());
  }
  ::aie::set_saturation(prev_sat);
  ::aie::set_rounding(prev_rnd);
  ::aie::store_v(dst, sv0.template to_vector<float>());
  ::aie::store_v(dst + 32, sv1.template to_vector<float>());
}

// sv_head for a block where some live pv has an exponent field below 24,
// whose v * ml * 2^(e - 150) can be denormal: |v| * pv rounded once as
// n = |v| * m, exact in acc32 as (|v| << 8) * (m >> 8) + |v| * (m & 0xff),
// then to_float(n, 150 - e), which rounds it to nearest-even (checked on
// NPU2 for every |v| and every m). Below exponent field 24 to_float's halves
// would be denormal, so n is converted 2^64 higher and brought down by its
// exponent field; |v| >= 1 keeps the product normal. pv is the head's bits.
__attribute__((noinline)) static void sv_head_low(const int8_t *v, int rows,
                                                  int n_rows, const int32_t *pv,
                                                  float *dst) {
  ::aie::vector<float, 32> sv_lo = ::aie::zeros<float, 32>();
  ::aie::vector<float, 32> sv_hi = ::aie::zeros<float, 32>();
#pragma clang loop min_iteration_count(16)
  for (int r = 0; r < rows; r++) {
    ::aie::vector<int16, kHD> v16 =
        ::aie::load_unaligned_v<kHD>(v + r * kRowBytes).unpack();
    ::aie::vector<uint16, kHD> av = ::aie::abs(v16).template cast_to<uint16>();
    ::aie::vector<uint16, kHD> av8 = ::aie::upshift(av, 8);
    const int32_t b = pv[r];
    const int32_t mant =
        r < n_rows && b > 0x7fffff ? (b & 0x7fffff) | 0x800000 : 0;
    const uint16_t mh = mant >> 8, ml = mant & 0xff;
    const int32_t e = r < n_rows ? b >> 23 : 127;
    const int sh = e < 24 ? 86 - e : 150 - e;
    const int32_t off = e < 24 ? 64 << 23 : 0;
    ::aie::accum<acc32, 32> n0 = ::aie::mul(av8.template extract<32>(0), mh);
    ::aie::accum<acc32, 32> n1 = ::aie::mul(av8.template extract<32>(1), mh);
    n0 = ::aie::mac(n0, av.template extract<32>(0), ml);
    n1 = ::aie::mac(n1, av.template extract<32>(1), ml);
    ::aie::vector<float, 32> f0 =
        ::aie::max(::aie::sub(::aie::to_float<float>(
                                  n0.template to_vector<int32>(0), sh)
                                  .template cast_to<int32>(),
                              off),
                   int32(0))
            .template cast_to<float>();
    ::aie::vector<float, 32> f1 =
        ::aie::max(::aie::sub(::aie::to_float<float>(
                                  n1.template to_vector<int32>(0), sh)
                                  .template cast_to<int32>(),
                              off),
                   int32(0))
            .template cast_to<float>();
    sv_lo = ::aie::add(
        sv_lo, ::aie::select(f0, ::aie::neg(f0),
                             ::aie::lt(v16.template extract<32>(0), int16(0))));
    sv_hi = ::aie::add(
        sv_hi, ::aie::select(f1, ::aie::neg(f1),
                             ::aie::lt(v16.template extract<32>(1), int16(0))));
  }
  ::aie::store_v(dst, sv_lo);
  ::aie::store_v(dst + 32, sv_hi);
}

// Each Q head's sv in state, fp32, to out: kREP*kHD floats, head by head.
static void copy_sv(int8_t *state, int8_t *out) {
  for (int h = 0; h < kREP; h++)
    copy_words<kHD * 4>(out + h * kHD * 4,
                        reinterpret_cast<int8_t *>(head_record(state, h) + 16));
}

extern "C" {

void llama_attn_state_init(int8_t *restrict state) {
  zero_words<kStateBytes>(state);
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
  int32_t *pv = expq;
  alignas(64) int32_t pv_m[kREP * kTB];
  alignas(64) uint16_t pv_kh[kREP * kTB];
  alignas(64) uint16_t pv_kl[kREP * kTB];
  alignas(64) float sv_b[kREP * kHD];

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
  // n_rows read as -1e30 so they never win the max and get e = 0; then
  // e = e^(score - max).
  alignas(64) float lanes[2 * kLanes] = {};
  ::aie::vector<float, kLanes> floor_v =
      ::aie::broadcast<float, kLanes>(-1e30f);
  for (int h = 0; h < kREP; h++) {
    const float q_scale = reinterpret_cast<const float *>(tail)[2 * h];
    ::aie::vector<float, kLanes> q_v = ::aie::broadcast<float, kLanes>(q_scale);
    ::aie::vector<float, kLanes> max_v = floor_v;
    for (int g = 0; g < groups; g++) {
      ::aie::vector<int32, kLanes> x =
          vec_mul_rne(q_v, ::aie::load_v<kLanes>(kslot + g * kLanes)
                               .template cast_to<float>())
              .template cast_to<int32>();
      // vec_mul_rne(x, 1/sqrt(hd)) for every x: its exponent less 3, or a
      // signed zero below FLT_MIN.
      ::aie::vector<float, kLanes> qk =
          ::aie::select(
              ::aie::bit_and(int32(0x80000000), x),
              ::aie::sub(x, int32(3 << 23)),
              ::aie::gt(::aie::bit_and(int32(0x7f800000), x), int32(3 << 23)))
              .template cast_to<float>();
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
    // Two groups a call; a group past `groups` is never read.
    ::aie::vector<float, 2 * kLanes> m_v =
        ::aie::broadcast<float, 2 * kLanes>(m_b);
    for (int g = 0; g < groups; g += 2)
      ::aie::store_v(expq + h * kTB + g * kLanes,
                     vec_exp_nonpos(::aie::sub(::aie::load_v<2 * kLanes>(
                                                   scores + g * kLanes)
                                                   .template cast_to<float>(),
                                               m_v))
                         .template cast_to<int32>());
  }

  // The sums run in row order with a head per lane, on 16-row groups
  // transposed by two zip stages.
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
  ::aie::store_v(inv_a, vec_recip_rne(sum_v));
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

  // pv = (e * inv_sum) * v_slot, per head and row, as sv_head takes it (a
  // zero significand past n_rows and for a zero or denormal pv, whose
  // products mul_rne flushes).
  ::aie::vector<int32, kLanes> pv_min =
      ::aie::broadcast<int32, kLanes>(0x7fffffff);
  ::aie::vector<int32, kLanes> pv_max = ::aie::zeros<int32, kLanes>();
  ::aie::vector<int32, kLanes> e_min = ::aie::broadcast<int32, kLanes>(255);
  for (int h = 0; h < kREP; h++) {
    ::aie::vector<float, kLanes> inv_v =
        ::aie::broadcast<float, kLanes>(inv_a[h]);
    for (int g = 0; g < groups; g++) {
      const int at = h * kTB + g * kLanes;
      ::aie::vector<int32, kLanes> b =
          vec_mul_rne(
              vec_mul_rne(
                  ::aie::load_v<kLanes>(expq + at).template cast_to<float>(),
                  inv_v),
              ::aie::load_v<kLanes>(vslot + g * kLanes)
                  .template cast_to<float>())
              .template cast_to<int32>();
      ::aie::store_v(pv + at, b);
      auto is_live =
          ::aie::lt(::aie::load_v<kLanes>(kLaneIndex),
                    ::aie::broadcast<int32, kLanes>(n_rows - g * kLanes));
      ::aie::vector<int32, kLanes> live = ::aie::select(
          ::aie::broadcast<int32, kLanes>(0x3f800000), b, is_live);
      pv_min = ::aie::min(pv_min, live);
      pv_max = ::aie::max(pv_max, live);
      auto has = is_live & ::aie::gt(b, int32(0x7fffff));
      ::aie::vector<int32, kLanes> mant = ::aie::select(
          ::aie::zeros<int32, kLanes>(),
          ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), b)),
          has);
      ::aie::vector<int32, kLanes> e =
          ::aie::downshift(::aie::bit_and(int32(0x7f800000), live), 23);
      e_min = ::aie::min(
          e_min, ::aie::select(::aie::broadcast<int32, kLanes>(255), e, has));
      ::aie::store_v(pv_m + at, mant);
      ::aie::accum<acc32, kLanes> t;
      // to_float's magic numbers for shift 150 - e, bf16 0x5301 - 128 * shift
      // and 0x4b01 - 128 * shift, doubled.
      ::aie::vector<int32, kLanes> e8 = ::aie::upshift(e, 8);
      t.from_vector(::aie::add(e8, int32(2 * (0x5301 - 128 * 150))));
      ::aie::store_v(pv_kh + at, t.template to_vector<uint16>(0));
      t.from_vector(::aie::add(e8, int32(2 * (0x4b01 - 128 * 150))));
      ::aie::store_v(pv_kl + at, t.template to_vector<uint16>(0));
    }
  }
  // Every pv non-negative with its exponent field at most 238; sv_head also
  // needs every nonzero one's at least 24.
  const bool fast =
      ::aie::reduce_min(pv_min) >= 0 && ::aie::reduce_max(pv_max) < (239 << 23);
  const bool low = ::aie::reduce_min(e_min) < 24;

  // sv = sum over rows of pv[r] * v[r], in row order, per head into its
  // record or, to be folded in, into sv_b.
  float *dst = count == 0 ? nullptr : sv_b;
  if (fast && !low) {
    for (int h = 0; h < kREP; h++)
      sv_head(block + kVOff, groups * kLanes, pv_m + h * kTB, pv_kh + h * kTB,
              pv_kl + h * kTB,
              dst ? dst + h * kHD : head_record(state, h) + 16);
  } else if (fast) {
    for (int h = 0; h < kREP; h++)
      sv_head_low(block + kVOff, groups * kLanes, n_rows, pv + h * kTB,
                  dst ? dst + h * kHD : head_record(state, h) + 16);
  } else {
    for (int h = 0; h < kREP; h++) {
      float *d = dst ? dst + h * kHD : head_record(state, h) + 16;
      for (int j = 0; j < kSlices; j++)
        ::aie::store_v(d + j * kLanes, ::aie::zeros<float, kLanes>());
      for (int r = 0; r < n_rows; r++) {
        ::aie::vector<float, kLanes> pv_v = ::aie::broadcast<float, kLanes>(
            __builtin_bit_cast(float, pv[h * kTB + r]));
        const int8_t *v = block + r * kRowBytes + kVOff;
#pragma clang loop unroll(disable)
        for (int j = 0; j < kSlices; j++)
          ::aie::store_v(
              d + j * kLanes,
              ::aie::add(::aie::load_v<kLanes>(d + j * kLanes),
                         vec_mul_rne(::aie::to_float<float>(
                                         ::aie::load_unaligned_v<kLanes>(
                                             v + j * kLanes)),
                                     pv_v)));
      }
    }
  }
  if (count > 0)
    for (int h = 0; h < kREP; h++)
      fold_sv(head_record(state, h) + 16, sv_b + h * kHD, w[h], w[kREP + h]);

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
  copy_words<kChunkBytes>(chunks + r * kChunkStride, chunk);
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
    copy_words<kRowBytes>(block_i32 + (row0 + r) * (kRowBytes / 4),
                          chunk + kQChunkBytes);
  llama_attn_block(chunk, block_i32, states + r * kStateBytes,
                   row0 >= 0 ? row0 + r + 1 : kTB);
}

void llama_attn_put_q(int8_t *restrict states, int8_t *restrict part,
                      int32_t r) {
  copy_words<kStateBytes>(part, states + r * kStateBytes);
}

// Folds two cores' states, the first core's first, in place into the first
// non-empty one, then writes each Q head's sv: out is kREP*kHD fp32, which
// the o projection's input quantizes once with the other heads'. The second
// core's state for query r is at states + r * kStateBytes.
void llama_attn_merge_pair_q(int8_t *restrict state0, int8_t *restrict states,
                             int8_t *restrict out, int32_t r) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  int8_t *state1 = states + r * kStateBytes;
  int32_t count0, count1;
  __builtin_memcpy(&count0, state0, 4);
  __builtin_memcpy(&count1, state1, 4);
  if (count0 > 0 && count1 > 0)
    fold_part(state0, state1);
  copy_sv(count0 > 0 ? state0 : state1, out);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

} // extern "C"
