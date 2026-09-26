//===- layer_norm_aie2p.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/scalar_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int N>
void layer_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  constexpr int H = N / 2;

  // cols is non-negative, so the unsigned divide lowers to a shift.
  const int vector_chunks = (uint32_t)cols / N;
  // A row of an odd number of H-lane halves ends in one half vector.
  const int tail = vector_chunks * N;
  const bool half = N == 32 && (cols & H);

  // Reduce the row sum in an f32 accumulator, not a bf16 vector: a bf16 running
  // sum drops low-order bits as the reduction length grows (embedding_dim is
  // typically thousands), so the mean -- and every quantity derived from it --
  // is already lossy before the variance is computed. The sum of squares is
  // already reduced in f32.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  ::aie::accum<accfloat, N> sum_sq_acc = ::aie::zeros<accfloat, N>();
  if (vector_chunks > 0) {
    const T *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(p);
      sum_acc = ::aie::add(sum_acc, reg_a);
      sum_sq_acc = ::aie::mac_square(sum_sq_acc, reg_a);
      p += N;
    }
  }
  if (half) {
    ::aie::vector<T, N> reg_a =
        ::aie::concat(::aie::load_v<H>(input + tail), ::aie::zeros<T, H>());
    sum_acc = ::aie::add(sum_acc, reg_a);
    sum_sq_acc = ::aie::mac_square(sum_sq_acc, reg_a);
  }

  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));
  float mean = scalar_mul(
      ::aie::reduce_add(sum_acc.template to_vector<float>()), inv_cols);
  float variance = scalar_mul_sub(
      scalar_mul(::aie::reduce_add(sum_sq_acc.template to_vector<float>()),
                 inv_cols),
      mean, mean);
  float inv_std = scalar_invsqrt(variance + epsilon);

  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>((T)mean);
  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>((T)inv_std);

  // gamma = 1 and beta = 0 here, so the affine pair is not applied at all.
  if (vector_chunks > 0) {
    const T *restrict pi = input;
    T *restrict po = output;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
      ::aie::store_v(po, ::aie::mul(diff_v, inv_std_v).template to_vector<T>());
      pi += N;
      po += N;
    }
  }
  if (half) {
    ::aie::vector<T, N> diff_v = ::aie::sub(
        ::aie::concat(::aie::load_v<H>(input + tail), ::aie::zeros<T, H>()),
        mean_v);
    ::aie::store_v(output + tail, ::aie::mul(diff_v, inv_std_v)
                                      .template to_vector<T>()
                                      .template extract<H>(0));
  }
  event1();
}

#if AIE_TUNED_AIE2P
// 64 lanes: a 32-lane bf16 mac takes half of the AIE2P multiplier.
constexpr unsigned kLanes = 64;
constexpr unsigned kPart = 16;
using bf16xN = ::aie::vector<bfloat16, kLanes>;
using f32_acc = ::aie::accum<accfloat, kLanes>;

// Two bf16 limbs of v in every lane, high first.
static inline void split2(float v, bf16xN &hi, bf16xN &lo) {
  f32_acc a;
  a.from_vector(::aie::broadcast<float, kLanes>(v));
  hi = a.to_vector<bfloat16>();
  lo = ::aie::msc(a, hi, ::aie::broadcast<bfloat16, kLanes>(1.0f))
           .to_vector<bfloat16>();
}

// The last cols % 64 of a row, in 16-lane parts over zeros.
static inline bf16xN load_tail(const bfloat16 *p, unsigned parts) {
  bf16xN v = ::aie::zeros<bfloat16, kLanes>();
  v.insert(0, ::aie::load_v<kPart>(p));
  if (parts > 1)
    v.insert(1, ::aie::load_v<kPart>(p + kPart));
  if (parts > 2)
    v.insert(2, ::aie::load_v<kPart>(p + 2 * kPart));
  return v;
}

static inline void store_tail(bfloat16 *p, bf16xN v, unsigned parts) {
  ::aie::store_v(p, v.extract<kPart>(0));
  if (parts > 1)
    ::aie::store_v(p + kPart, v.extract<kPart>(1));
  if (parts > 2)
    ::aie::store_v(p + 2 * kPart, v.extract<kPart>(2));
}

// x s + c with s = s_hi + s_lo = inv_std and c = -mean s, both in f32, so
// that the output is rounded to bf16 once, as on AIE2.
static void layer_norm_aie2p(const bfloat16 *restrict input,
                             bfloat16 *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const unsigned chunks = (uint32_t)cols / kLanes;
  const unsigned tail = chunks * kLanes;
  const unsigned parts = ((uint32_t)cols - tail) / kPart;

  const bf16xN one = ::aie::broadcast<bfloat16, kLanes>(1.0f);
  f32_acc sum = ::aie::zeros<accfloat, kLanes>();
  f32_acc sum_sq = ::aie::zeros<accfloat, kLanes>();
  if (chunks > 0) {
    const bfloat16 *restrict p = input;
    AIE_LOOP_UNROLL(4)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < chunks; i++) {
      bf16xN x = ::aie::load_v<kLanes>(p);
      sum = ::aie::mac(sum, x, one);
      sum_sq = ::aie::mac_square(sum_sq, x);
      p += kLanes;
    }
  }
  if (parts > 0) {
    bf16xN x = load_tail(input + tail, parts);
    sum = ::aie::mac(sum, x, one);
    sum_sq = ::aie::mac_square(sum_sq, x);
  }

  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));
  float mean = scalar_mul(::aie::reduce_add(sum.to_vector<float>()), inv_cols);
  float variance = scalar_mul_sub(
      scalar_mul(::aie::reduce_add(sum_sq.to_vector<float>()), inv_cols), mean,
      mean);
  bf16xN s_hi, s_lo, m_hi, m_lo;
  split2(scalar_invsqrt(variance + epsilon), s_hi, s_lo);
  split2(mean, m_hi, m_lo);
  f32_acc c = ::aie::negmul(m_lo, s_lo);
  c = ::aie::msc(c, m_lo, s_hi);
  c = ::aie::msc(c, m_hi, s_lo);
  c = ::aie::msc(c, m_hi, s_hi);

  if (chunks > 0) {
    const bfloat16 *restrict pi = input;
    bfloat16 *restrict po = output;
    AIE_LOOP_UNROLL(4)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < chunks; i++) {
      bf16xN x = ::aie::load_v<kLanes>(pi);
      ::aie::store_v(
          po,
          ::aie::mac(::aie::mac(c, x, s_lo), x, s_hi).to_vector<bfloat16>());
      pi += kLanes;
      po += kLanes;
    }
  }
  if (parts > 0) {
    bf16xN x = load_tail(input + tail, parts);
    store_tail(
        output + tail,
        ::aie::mac(::aie::mac(c, x, s_lo), x, s_hi).to_vector<bfloat16>(),
        parts);
  }
  event1();
}
#endif

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  // One bf16 multiply's lanes, the multiple the factory holds cols to.
  // conv_even rounding matches the reference math more closely than the
  // default floor mode for the normalize pass.
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
#if AIE_TUNED_AIE2P
  layer_norm_aie2p(input, output, cols);
#else
  layer_norm<bfloat16, AIE_BF16_LANES>(input, output, cols);
#endif
  ::aie::set_rounding(saved_rounding);
}
}
