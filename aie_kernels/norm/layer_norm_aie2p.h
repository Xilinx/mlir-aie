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

  // cols is non-negative, so the unsigned divide lowers to a shift.
  const int vector_chunks = (uint32_t)cols / N;

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
  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  // N=32 bf16 = 512 bits = one AIE2P vector register.  conv_even rounding
  // matches the reference math more closely than the default floor mode for
  // the normalize pass.
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  layer_norm<bfloat16, 32>(input, output, cols);
}
}
