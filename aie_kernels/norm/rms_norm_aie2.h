//===- rms_norm_aie2.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/bf16_limbs.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 1 / sqrt(sum_sq / cols + epsilon) as [hi | lo] limbs, to about 1e-5.
static inline v32bfloat16 inv_rms_limbs(float sum_sq, int32_t cols,
                                        float epsilon) {
  // A row fits in 64 KB, so cols < 2^15 and the quotient keeps 17+ bits.
  int32_t inv_cols_q31 = (int32_t)(0x7fffffffu / (uint32_t)cols);
  v32bfloat16 inv_cols =
      limbs(broadcast_to_v16accfloat(::aie::to_float<float>(inv_cols_q31, 31)));
  v32bfloat16 s = limbs(broadcast_to_v16accfloat(sum_sq));
  v16accfloat m_acc = mac_elem_16_2(
      s, bf16_pair(limb(inv_cols, 1), limb(inv_cols, 1)),
      mac_elem_16_2(s, bf16_pair(limb(inv_cols, 0), limb(inv_cols, 0)),
                    broadcast_to_v16accfloat(epsilon)));
  return inv_sqrt_limbs(m_acc);
}

// x * inv_rms for one chunk; each half against [hi | lo] is one mac.
template <typename T, int N>
static inline void scale_chunk(const T *restrict in, T *restrict out,
                               v32bfloat16 inv_rms) {
  v16bfloat16 x0 = ::aie::load_v<N / 2>(in);
  v16bfloat16 x1 = ::aie::load_v<N / 2>(in + N / 2);
  ::aie::store_v(out, ::aie::vector<T, N / 2>(to_v16bfloat16(
                          mul_elem_16_2(bf16_pair(x0, x0), inv_rms))));
  ::aie::store_v(out + N / 2, ::aie::vector<T, N / 2>(to_v16bfloat16(
                                  mul_elem_16_2(bf16_pair(x1, x1), inv_rms))));
}

template <typename T, int N>
void rms_norm(const T *restrict input, T *restrict output, int32_t cols,
              float epsilon = 1e-5f) {
  static_assert(N == 32, "the sum of squares packs 32 lanes per mac");
  event0();
  const unsigned vector_chunks = (uint32_t)cols / N;
  const int remaining = cols - vector_chunks * N;
  // See MIN_CHUNKS in layer_norm_aie2.h.
  constexpr unsigned MIN_CHUNKS = 8;
  const bool pipelined = vector_chunks >= MIN_CHUNKS;

  // aie::mac_square pads the upper 16 lanes of each bf16 mac with zeros. Fed
  // the row in both operands, a mac sums two squares into each of its 16 f32
  // lanes, so a 32-lane chunk takes one mac instead of two. Four accumulators
  // split the mac latency chain.
  v16accfloat sq_acc0 = ::aie::zeros<accfloat, 16>();
  v16accfloat sq_acc1 = ::aie::zeros<accfloat, 16>();
  v16accfloat sq_acc2 = ::aie::zeros<accfloat, 16>();
  v16accfloat sq_acc3 = ::aie::zeros<accfloat, 16>();
  unsigned chunk = 0;
  if (pipelined) {
    const T *restrict p = input;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS / 4)
    for (unsigned i = 0; i < vector_chunks / 4; i++) {
      v32bfloat16 x0 = ::aie::load_v<N>(p);
      v32bfloat16 x1 = ::aie::load_v<N>(p + N);
      v32bfloat16 x2 = ::aie::load_v<N>(p + 2 * N);
      v32bfloat16 x3 = ::aie::load_v<N>(p + 3 * N);
      sq_acc0 = mac_elem_16_2(x0, x0, sq_acc0);
      sq_acc1 = mac_elem_16_2(x1, x1, sq_acc1);
      sq_acc2 = mac_elem_16_2(x2, x2, sq_acc2);
      sq_acc3 = mac_elem_16_2(x3, x3, sq_acc3);
      p += 4 * N;
    }
    chunk = vector_chunks & ~3u;
  }
  for (; chunk < vector_chunks; chunk++) {
    v32bfloat16 x = ::aie::load_v<N>(input + chunk * N);
    sq_acc0 = mac_elem_16_2(x, x, sq_acc0);
  }
  float sum_sq = ::aie::reduce_add(
      ::aie::add(::aie::add(::aie::accum<accfloat, 16>(sq_acc0),
                            ::aie::accum<accfloat, 16>(sq_acc1)),
                 ::aie::add(::aie::accum<accfloat, 16>(sq_acc2),
                            ::aie::accum<accfloat, 16>(sq_acc3)))
          .template to_vector<float>());

  if (remaining > 0) {
    int start_idx = vector_chunks * N;
    for (int i = 0; i < remaining; i++) {
      T val = input[start_idx + i];
      float square = static_cast<float>(val) * static_cast<float>(val);
      sum_sq += square;
    }
  }

  v32bfloat16 inv_rms = inv_rms_limbs(sum_sq, cols, epsilon);

  if (pipelined) {
    const T *restrict pi = input;
    T *restrict po = output;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS)
    for (unsigned i = 0; i < vector_chunks; i++) {
      scale_chunk<T, N>(pi, po, inv_rms);
      pi += N;
      po += N;
    }
  } else {
    for (unsigned i = 0; i < vector_chunks; i++)
      scale_chunk<T, N>(input + i * N, output + i * N, inv_rms);
  }

  if (remaining > 0) {
    ::aie::vector<T, N> inv_rms_v(inv_rms);
    float inv_rms_f =
        static_cast<float>(inv_rms_v[0]) + static_cast<float>(inv_rms_v[N / 2]);
    int start_idx = vector_chunks * N;
    for (int i = 0; i < remaining; i++) {
      T val = input[start_idx + i];
      output[start_idx + i] =
          static_cast<T>(static_cast<float>(val) * inv_rms_f);
    }
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  rms_norm<bfloat16, 32>(input, output, cols);
  ::aie::set_rounding(saved_rounding);
}

void rms_norm_eps(bfloat16 *input, bfloat16 *output, int32_t cols,
                  float epsilon) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  rms_norm<bfloat16, 32>(input, output, cols, epsilon);
  ::aie::set_rounding(saved_rounding);
}
}
