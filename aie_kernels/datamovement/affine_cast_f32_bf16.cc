//===- affine_cast_f32_bf16.cc ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <cassert>
#include <stdint.h>

// out = bf16(in * gamma + beta) over a row-major rows x cols tile, with gamma
// and beta per column. `gb` holds gamma then beta, `cols` values each, so the
// pair needs one input DMA channel rather than two.
//
// The multiply and the add round separately, as the host reference does, and
// the bf16 store rounds with conv_even, as cast_f32_bf16.cc does.
template <int N>
void affine_cast_f32_bf16(const float *restrict input, const float *restrict gb,
                          bfloat16 *restrict output, int32_t rows,
                          int32_t cols) {
  assert(cols % N == 0);
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *restrict pg = gb;
  const float *restrict pb = gb + cols;
  for (int j = 0; j < cols; j += N, pg += N, pb += N) {
    ::aie::vector<float, N> gamma = ::aie::load_v<N>(pg);
    ::aie::vector<float, N> beta = ::aie::load_v<N>(pb);
    const float *restrict in = input + j;
    bfloat16 *restrict out = output + j;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < rows; i++, in += cols, out += cols) {
      ::aie::vector<float, N> scaled =
          ::aie::mul(::aie::load_v<N>(in), gamma).template to_vector<float>();
      ::aie::accum<accfloat, N> a;
      a.from_vector(::aie::add(scaled, beta));
      ::aie::store_v(out, a.template to_vector<bfloat16>());
    }
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void affine_cast_f32_bf16(float *input, float *gb, bfloat16 *output,
                          int32_t rows, int32_t cols) {
  affine_cast_f32_bf16<16>(input, gb, output, rows, cols);
}
}
