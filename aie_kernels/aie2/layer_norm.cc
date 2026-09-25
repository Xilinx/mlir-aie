//===- layernorm.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdlib.h>

// The row statistics run on the vector unit, every lane holding the same
// value: scalar f32 mul, div and int-to-float are soft-float libcalls on AIE2,
// and its f32 vector multiply is no better (see activations.h). A bf16 mac
// sums two exact products into each f32 lane, lane i getting a[i] b[i] +
// a[i+16] b[i+16], so an f32 held as bf16 limbs times a bf16 is one mac per two
// limbs.
static inline v32bfloat16 bf16_pair(v16bfloat16 lo, v16bfloat16 hi) {
  return concat(lo, hi);
}

static inline v16bfloat16 bf16_lanes(int32_t bits) {
  return extract_v16bfloat16(
      broadcast_to_v32bfloat16(__builtin_bit_cast(bfloat16, (int16_t)bits)), 0);
}

// x - hi, exactly.
static inline v16accfloat residual(v16accfloat x, v16bfloat16 hi) {
  return msc_elem_16_2(bf16_pair(hi, bf16_lanes(0)),
                       broadcast_one_to_v32bfloat16(), x);
}

// [hi | lo] limbs of x; hi + lo holds its top 16 bits.
static inline v32bfloat16 limbs(v16accfloat x) {
  v16bfloat16 hi = to_v16bfloat16(x);
  return bf16_pair(hi, to_v16bfloat16(residual(x, hi)));
}

static inline v16bfloat16 limb(v32bfloat16 x, int i) {
  return extract_v16bfloat16(x, i);
}

// 1 / sqrt(m) as [hi | lo] limbs, to about 1e-5, for a normal m > 0.
static inline v32bfloat16 inv_sqrt_limbs(v16accfloat m_acc) {
  v32bfloat16 m = limbs(m_acc);

  // y0: the bit-trick estimate, cut to bf16 (4% off).
  float m_f = ::aie::vector<float, 16>(v16float(m_acc))[0];
  int32_t y0_bits =
      (0x5f3759df - (__builtin_bit_cast(int32_t, m_f) >> 1)) >> 16;
  v16bfloat16 y0 = bf16_lanes(y0_bits);
  v16bfloat16 zero = bf16_lanes(0);

  // y1 = y0 + y0 / 2 (1 - m y0^2), a Newton step in bf16 (5e-3 off).
  v16bfloat16 m_y0 = to_v16bfloat16(mul_elem_16_2(m, bf16_pair(y0, y0)));
  v16accfloat r = msc_elem_16_2(bf16_pair(m_y0, zero), bf16_pair(y0, zero),
                                broadcast_to_v16accfloat(1.0f));
  v16bfloat16 y1 = to_v16bfloat16(mac_elem_16_2(
      bf16_pair(to_v16bfloat16(r), zero),
      bf16_pair(bf16_lanes(y0_bits - 0x80), zero), ups_to_v16accfloat(y0)));

  // y = y1 (1 + r / 2 + 3 r^2 / 8) with r = 1 - m y1^2 taken exactly: y1^2 is
  // exact in two limbs, and |r| < 1e-2 leaves the cubic term under 4e-7.
  bfloat16 y1_s = ::aie::vector<bfloat16, 16>(y1)[0];
  int32_t y1_bits = __builtin_bit_cast(int16_t, y1_s);
  v32bfloat16 y1_sq =
      limbs(mul_elem_16_2(bf16_pair(y1, zero), bf16_pair(y1, zero)));
  r = msc_elem_16_2(m, bf16_pair(limb(y1_sq, 1), limb(y1_sq, 1)),
                    broadcast_to_v16accfloat(1.0f));
  r = msc_elem_16_2(m, bf16_pair(limb(y1_sq, 0), limb(y1_sq, 0)), r);
  v32bfloat16 r_limbs = limbs(r);
  v16bfloat16 r_sq = to_v16bfloat16(mul_elem_16_2(
      bf16_pair(limb(r_limbs, 0), zero), bf16_pair(limb(r_limbs, 0), zero)));
  v16bfloat16 y1_3_8 = to_v16bfloat16(mul_elem_16_2(
      bf16_pair(y1, zero), bf16_pair(bf16_lanes(0x3ec0), zero))); // 0.375
  v16bfloat16 y1_half = bf16_lanes(y1_bits - 0x80);
  v16accfloat y = mac_elem_16_2(r_limbs, bf16_pair(y1_half, y1_half),
                                ups_to_v16accfloat(y1));
  y = mac_elem_16_2(bf16_pair(r_sq, zero), bf16_pair(y1_3_8, zero), y);
  return limbs(y);
}

// (x - mean) inv_std for 16 lanes as x hi + x lo + c, c = -mean inv_std.
template <typename T>
static inline void normalize_16(const T *restrict in, T *restrict out,
                                v32bfloat16 inv_std, v16accfloat c) {
  v16bfloat16 x = ::aie::load_v<16>(in);
  ::aie::store_v(out, ::aie::vector<T, 16>(to_v16bfloat16(
                          mac_elem_16_2(bf16_pair(x, x), inv_std, c))));
}

template <typename T>
void layer_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr unsigned N = 32;
  // cols is a multiple of 16: whole 32-lane chunks, then maybe one half.
  const unsigned vector_chunks = (uint32_t)cols / N;
  const bool half_chunk = (uint32_t)cols & (N / 2);
  // The pipelined loops are promised MIN_CHUNKS chunks, since the promised
  // trip count caps the stage count. Shorter rows take the plain loops.
  constexpr unsigned MIN_CHUNKS = 8;
  const bool pipelined = vector_chunks >= MIN_CHUNKS;
  // 1 / n from a Q31 quotient; a row fits in 64 KB, so it keeps 17+ bits.
  // Divided first: no vector state is live across the call.
  const int32_t inv_n_q31 = (int32_t)(0x80000000u / (uint32_t)cols);

  // Sum and sum of squares, each product exact and summed in f32: against
  // ones a mac adds two elements into each of its 16 lanes, against itself
  // two squares.
  const v32bfloat16 ones = broadcast_one_to_v32bfloat16();
  v16accfloat sum0 = ::aie::zeros<accfloat, 16>();
  v16accfloat sum1 = ::aie::zeros<accfloat, 16>();
  v16accfloat sq0 = ::aie::zeros<accfloat, 16>();
  v16accfloat sq1 = ::aie::zeros<accfloat, 16>();
  unsigned chunk = 0;
  if (pipelined) {
    const T *restrict p = input;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS / 2)
    for (unsigned i = 0; i < vector_chunks / 2; i++) {
      v32bfloat16 x = ::aie::load_v<N>(p);
      v32bfloat16 y = ::aie::load_v<N>(p + N);
      sum0 = mac_elem_16_2(x, ones, sum0);
      sq0 = mac_elem_16_2(x, x, sq0);
      sum1 = mac_elem_16_2(y, ones, sum1);
      sq1 = mac_elem_16_2(y, y, sq1);
      p += 2 * N;
    }
    chunk = vector_chunks & ~1u;
  }
  for (; chunk < vector_chunks; chunk++) {
    v32bfloat16 x = ::aie::load_v<N>(input + chunk * N);
    sum0 = mac_elem_16_2(x, ones, sum0);
    sq0 = mac_elem_16_2(x, x, sq0);
  }
  if (half_chunk) {
    v32bfloat16 x = bf16_pair(::aie::load_v<N / 2>(input + vector_chunks * N),
                              bf16_lanes(0));
    sum0 = mac_elem_16_2(x, ones, sum0);
    sq0 = mac_elem_16_2(x, x, sq0);
  }
  v16accfloat s1 = broadcast_to_v16accfloat(
      ::aie::reduce_add(::aie::add(::aie::accum<accfloat, 16>(sum0),
                                   ::aie::accum<accfloat, 16>(sum1))
                            .template to_vector<float>()));
  v16accfloat s2 = broadcast_to_v16accfloat(
      ::aie::reduce_add(::aie::add(::aie::accum<accfloat, 16>(sq0),
                                   ::aie::accum<accfloat, 16>(sq1))
                            .template to_vector<float>()));

  // var = (n s2 - s1^2) / n^2 keeps the cancellation in one f32 accumulator.
  // n < 2^16 is exact in two limbs; s1 and s2 take three.
  v16bfloat16 zero = bf16_lanes(0);
  v32bfloat16 n =
      limbs(broadcast_to_v16accfloat(::aie::to_float<float>(cols, 0)));
  v16bfloat16 n0 = limb(n, 0), n1 = limb(n, 1);
  v16bfloat16 a0 = to_v16bfloat16(s2);
  v16accfloat a_r = residual(s2, a0);
  v16bfloat16 a1 = to_v16bfloat16(a_r);
  v16bfloat16 a2 = to_v16bfloat16(residual(a_r, a1));
  v16bfloat16 b0 = to_v16bfloat16(s1);
  v16accfloat b_r = residual(s1, b0);
  v16bfloat16 b1 = to_v16bfloat16(b_r);
  v16bfloat16 b2 = to_v16bfloat16(residual(b_r, b1));
  v16accfloat d = mul_elem_16_2(bf16_pair(a0, a1), bf16_pair(n0, n0));
  d = mac_elem_16_2(bf16_pair(a2, a0), bf16_pair(n0, n1), d);
  d = mac_elem_16_2(bf16_pair(a1, a2), bf16_pair(n1, n1), d);
  // Doubling a bf16 is exact.
  v16bfloat16 b1_2 = to_v16bfloat16(
      mul_elem_16_2(bf16_pair(b1, zero), bf16_pair(bf16_lanes(0x4000), zero)));
  v16bfloat16 b2_2 = to_v16bfloat16(
      mul_elem_16_2(bf16_pair(b2, zero), bf16_pair(bf16_lanes(0x4000), zero)));
  d = msc_elem_16_2(bf16_pair(b0, b0), bf16_pair(b0, b1_2), d);
  d = msc_elem_16_2(bf16_pair(b0, b1), bf16_pair(b2_2, b1), d);
  d = msc_elem_16_2(bf16_pair(b1, b2), bf16_pair(b2_2, b2), d);

  v32bfloat16 inv_n =
      limbs(broadcast_to_v16accfloat(::aie::to_float<float>(inv_n_q31, 31)));
  v16bfloat16 i0 = limb(inv_n, 0), i1 = limb(inv_n, 1);
  v32bfloat16 d_limbs = limbs(d);
  v32bfloat16 d_n = limbs(mac_elem_16_2(
      d_limbs, bf16_pair(i1, i1), mul_elem_16_2(d_limbs, bf16_pair(i0, i0))));
  v16accfloat var_eps = mac_elem_16_2(
      d_n, bf16_pair(i1, i1),
      mac_elem_16_2(d_n, bf16_pair(i0, i0), broadcast_to_v16accfloat(1e-5f)));
  v32bfloat16 inv_std = inv_sqrt_limbs(var_eps);

  // mean = q + e: q = s1 inv_n to 16 bits, and e = (s1 - n q) inv_n corrects
  // it to f32, zero when n divides s1.
  v32bfloat16 s1_limbs = bf16_pair(b0, b1);
  v32bfloat16 q = limbs(mac_elem_16_2(
      s1_limbs, bf16_pair(i1, i1), mul_elem_16_2(s1_limbs, bf16_pair(i0, i0))));
  v16bfloat16 q0 = limb(q, 0), q1 = limb(q, 1);
  v16bfloat16 r0 = to_v16bfloat16(msc_elem_16_2(
      q, bf16_pair(n1, n1), msc_elem_16_2(q, bf16_pair(n0, n0), s1)));
  v16bfloat16 e =
      to_v16bfloat16(mul_elem_16_2(bf16_pair(r0, zero), bf16_pair(i0, zero)));
  v16accfloat c = msc_elem_16_2(
      inv_std, bf16_pair(q1, q1),
      msc_elem_16_2(inv_std, bf16_pair(q0, q0), ::aie::zeros<accfloat, 16>()));
  c = msc_elem_16_2(inv_std, bf16_pair(e, e), c);

  if (pipelined) {
    const T *restrict pi = input;
    T *restrict po = output;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS)
    for (unsigned i = 0; i < vector_chunks; i++) {
      normalize_16<T>(pi, po, inv_std, c);
      normalize_16<T>(pi + N / 2, po + N / 2, inv_std, c);
      pi += N;
      po += N;
    }
  } else {
    for (unsigned i = 0; i < 2 * vector_chunks; i++)
      normalize_16<T>(input + i * (N / 2), output + i * (N / 2), inv_std, c);
  }
  if (half_chunk)
    normalize_16<T>(input + vector_chunks * N, output + vector_chunks * N,
                    inv_std, c);
  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  // conv_even rounding matches the reference math more closely than the
  // default floor mode.
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  layer_norm<bfloat16>(input, output, cols);
  ::aie::set_rounding(saved_rounding);
}
}
