//===- rms_norm_aie2p.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/scalar_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int N>
void rms_norm(const T *restrict input, T *restrict output, int32_t cols,
              float epsilon = 1e-5f) {
  event0();
  // cols is non-negative, so the unsigned divide lowers to a shift/mask.
  const int vector_chunks = (uint32_t)cols / N;
  const int remaining = (uint32_t)cols % N;
  const int tail_start = vector_chunks * N;

  // A walking pointer, so the load can post-increment.
  ::aie::accum<accfloat, N> acc = ::aie::zeros<accfloat, N>();
  if (vector_chunks > 0) {
    const T *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      acc = ::aie::mac_square(acc, ::aie::load_v<N>(p));
      p += N;
    }
  }

  // Square the tail on the vector unit too: gather it into a zero-padded
  // register rather than doing scalar float math on each element.
  ::aie::vector<T, N> tail_v = ::aie::zeros<T, N>();
  if (remaining > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < remaining; i++)
      tail_v[i] = input[tail_start + i];
    acc = ::aie::mac_square(acc, tail_v);
  }

  const float sum_sq = ::aie::reduce_add(acc.template to_vector<float>());
  const float rms =
      scalar_mul(sum_sq, ::aie::inv(::aie::to_float<float>(cols))) + epsilon;
  const ::aie::vector<T, N> inv_rms_v =
      ::aie::broadcast<T, N>(static_cast<T>(scalar_invsqrt(rms)));

  if (vector_chunks > 0) {
    const T *restrict pi = input;
    T *restrict po = output;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::store_v(
          po,
          ::aie::mul(::aie::load_v<N>(pi), inv_rms_v).template to_vector<T>());
      pi += N;
      po += N;
    }
  }

  if (remaining > 0) {
    const ::aie::vector<T, N> out_v =
        ::aie::mul(tail_v, inv_rms_v).template to_vector<T>();
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < remaining; i++)
      output[tail_start + i] = out_v[i];
  }
  event1();
}

#if AIE_TUNED_AIE2P
// 64 lanes: a 32-lane bf16 mac takes half of the AIE2P multiplier.
constexpr unsigned kLanes = 64;
constexpr unsigned kPart = 16;
using bf16xN = ::aie::vector<bfloat16, kLanes>;
using f32_acc = ::aie::accum<accfloat, kLanes>;

// The last cols % 64 of a row over zeros: 16-lane parts, then single elements.
static inline bf16xN load_tail(const bfloat16 *p, unsigned parts,
                               unsigned rest) {
  bf16xN v = ::aie::zeros<bfloat16, kLanes>();
  if (parts > 0)
    v.insert(0, ::aie::load_v<kPart>(p));
  if (parts > 1)
    v.insert(1, ::aie::load_v<kPart>(p + kPart));
  if (parts > 2)
    v.insert(2, ::aie::load_v<kPart>(p + 2 * kPart));
  AIE_LOOP_MAX_ITERATION_COUNT(kPart - 1)
  for (unsigned i = 0; i < rest; i++)
    v[parts * kPart + i] = p[parts * kPart + i];
  return v;
}

static inline void store_tail(bfloat16 *p, bf16xN v, unsigned parts,
                              unsigned rest) {
  if (parts > 0)
    ::aie::store_v(p, v.extract<kPart>(0));
  if (parts > 1)
    ::aie::store_v(p + kPart, v.extract<kPart>(1));
  if (parts > 2)
    ::aie::store_v(p + 2 * kPart, v.extract<kPart>(2));
  AIE_LOOP_MAX_ITERATION_COUNT(kPart - 1)
  for (unsigned i = 0; i < rest; i++)
    p[parts * kPart + i] = v[parts * kPart + i];
}

// x s_hi + x s_lo with s_hi + s_lo = inv_rms in f32, so that the output is
// rounded to bf16 once.
static void rms_norm_aie2p(const bfloat16 *restrict input,
                           bfloat16 *restrict output, int32_t cols,
                           float epsilon) {
  event0();
  const unsigned chunks = (uint32_t)cols / kLanes;
  const unsigned tail = chunks * kLanes;
  const unsigned parts = ((uint32_t)cols - tail) / kPart;
  const unsigned rest = ((uint32_t)cols - tail) % kPart;

  f32_acc acc = ::aie::zeros<accfloat, kLanes>();
  if (chunks > 0) {
    const bfloat16 *restrict p = input;
    AIE_LOOP_UNROLL(4)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < chunks; i++) {
      acc = ::aie::mac_square(acc, ::aie::load_v<kLanes>(p));
      p += kLanes;
    }
  }
  bf16xN tail_v = ::aie::zeros<bfloat16, kLanes>();
  if (tail < (uint32_t)cols) {
    tail_v = load_tail(input + tail, parts, rest);
    acc = ::aie::mac_square(acc, tail_v);
  }

  const float sum_sq = ::aie::reduce_add(acc.to_vector<float>());
  const float inv_rms = scalar_invsqrt(
      scalar_mul(sum_sq, ::aie::inv(::aie::to_float<float>(cols))) + epsilon);
  f32_acc a;
  a.from_vector(::aie::broadcast<float, kLanes>(inv_rms));
  const bf16xN s_hi = a.to_vector<bfloat16>();
  const bf16xN s_lo =
      ::aie::msc(a, s_hi, ::aie::broadcast<bfloat16, kLanes>(1.0f))
          .to_vector<bfloat16>();

  if (chunks > 0) {
    const bfloat16 *restrict pi = input;
    bfloat16 *restrict po = output;
    AIE_LOOP_UNROLL(4)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < chunks; i++) {
      bf16xN x = ::aie::load_v<kLanes>(pi);
      ::aie::store_v(
          po, ::aie::mac(::aie::mul(x, s_lo), x, s_hi).to_vector<bfloat16>());
      pi += kLanes;
      po += kLanes;
    }
  }
  if (tail < (uint32_t)cols)
    store_tail(output + tail,
               ::aie::mac(::aie::mul(tail_v, s_lo), tail_v, s_hi)
                   .to_vector<bfloat16>(),
               parts, rest);
  event1();
}
#endif

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
#if AIE_TUNED_AIE2P
  rms_norm_aie2p(input, output, cols, 1e-5f);
#else
  // N=32 bf16 = 512 bits = one AIE2P vector register; the tail loop handles a
  // cols not divisible by 32.
  rms_norm<bfloat16, 32>(input, output, cols);
#endif
}

void rms_norm_eps(bfloat16 *input, bfloat16 *output, int32_t cols,
                  float epsilon) {
#if AIE_TUNED_AIE2P
  rms_norm_aie2p(input, output, cols, epsilon);
#else
  rms_norm<bfloat16, 32>(input, output, cols, epsilon);
#endif
}
}
