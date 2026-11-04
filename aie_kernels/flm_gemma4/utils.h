/*
Copyright (C) 2014-2022 Xilinx, Inc.
Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
    SPDX-License-Identifier: MIT
*/

#ifndef AIE_KERNELS_FLM_GEMMA4_UTILS_H
#define AIE_KERNELS_FLM_GEMMA4_UTILS_H

#include "../aie_kernel_utils.h" // AIE_LOOP_*

#include <aie_api/aie.hpp>
#include <cstdint>

using bf16 = bfloat16;

// Stores 256 bits at a time, so c needs only 32-byte alignment. A wider store
// to a buffer aligned below its width drops part of the store on AIE2P.
template <typename T, int M>
void zero_256(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8);
  static_assert((M) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

/// \brief Copy M elements between non-overlapping, 32-byte-aligned buffers.
/// Vector loads/stores allow pipelining; an opaque memcpy call prevents it.
template <typename T, int M>
void copy_vectorized(T *__restrict dst, const T *__restrict src) {
  constexpr int r = 256 / (sizeof(T) * 8);
  static_assert((M) % r == 0);
  const T *__restrict src_end = src + M;
  for (; src < src_end; src += r, dst += r) {
    aie::store_v(dst, aie::load_v<r>(src));
  }
}

template <int M>
void scale_vectorized(bf16 *y, const bf16 scale) {
  for (int i = 0; i < M / 16; i++) {
    aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
    aie::vector<bf16, 16> out_vec =
        aie::mul(y_vec, aie::broadcast<bf16, 16>(scale));
    aie::store_v(y, out_vec);
    y += 16;
  }
}

/// \brief Narrow N floats to bf16, allowing in-place conversion.
/// With the same base address, bf16 stores trail float loads. Do not use
/// restrict.
template <int N>
void narrow_to_bf16(bf16 *out, const float *in) {
  constexpr int vec_factor = 16;
  static_assert(N % vec_factor == 0);
  for (int i = 0; i < N / vec_factor; i++) {
    aie::accum<accfloat, vec_factor> acc;
    acc.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, acc.template to_vector<bf16>());
    in += vec_factor;
    out += vec_factor;
  }
}

// out[l] = the sum of x[32 l .. 32 l + 31], for the N / 32 groups of x: the
// column sums that fold a q4nx block's minima into one MAC per group.
template <int N>
void group_sums_32(bf16 *out, const bf16 *x) {
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < N / 32; l++) {
    out[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
  }
}

#endif // AIE_KERNELS_FLM_GEMMA4_UTILS_H
