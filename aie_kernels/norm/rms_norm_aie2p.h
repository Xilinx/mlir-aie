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

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  // N=32 bf16 = 512 bits = one AIE2P vector register; the tail loop handles a
  // cols not divisible by 32.
  rms_norm<bfloat16, 32>(input, output, cols);
}

void rms_norm_eps(bfloat16 *input, bfloat16 *output, int32_t cols,
                  float epsilon) {
  rms_norm<bfloat16, 32>(input, output, cols, epsilon);
}
}
