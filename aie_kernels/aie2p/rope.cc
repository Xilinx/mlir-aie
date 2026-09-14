//===- rope.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rope_kernel(const T *restrict input, const T *restrict lut,
                 T *restrict output, int32_t dims) {
  event0();

  for (int v = 0; v < dims; v += N) {
    ::aie::vector<T, N> x = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> cache = ::aie::load_v<N>(lut + v);

    // Extract even and odd elements
    ::aie::vector<T, N / 2> x_even = ::aie::filter_even(x, 1);
    ::aie::vector<T, N / 2> x_odd = ::aie::filter_odd(x, 1);
    ::aie::vector<T, N / 2> cos_val = ::aie::filter_even(cache, 1);
    ::aie::vector<T, N / 2> sin_val = ::aie::filter_odd(cache, 1);

    // Perform ROPE calculations
    ::aie::vector<T, N / 2> even_cos = ::aie::mul(x_even, cos_val);
    ::aie::vector<T, N / 2> even_sin = ::aie::mul(x_even, sin_val);
    ::aie::vector<T, N / 2> odd_cos = ::aie::mul(x_odd, cos_val);
    ::aie::vector<T, N / 2> odd_sin = ::aie::mul(x_odd, sin_val);

    ::aie::vector<T, N / 2> output_even = ::aie::sub(even_cos, odd_sin);
    ::aie::vector<T, N / 2> output_odd = ::aie::add(even_sin, odd_cos);

    auto [low, high] = ::aie::interleave_zip(output_even, output_odd, 1);
    ::aie::vector<T, N> y = ::aie::concat(low, high);
    ::aie::store_v(output + v, y);
  }
  event1();
}

// Two-halves RoPE (the layout used by HuggingFace transformers): the first and
// second halves of the vector are rotated against each other, rather than the
// even/odd interleave of the Llama-paper method in rope_kernel above.  Ported
// from IRON so designs targeting HF-style weights have a matching kernel.
template <typename T, int N>
void rope_kernel_two_halves(const T *restrict input, const T *restrict lut,
                            T *restrict output, int32_t dims) {
  event0();

  auto dims_half = dims / 2;
  for (int v = 0, i = 0; v < dims_half; v += N, i += 2 * N) {
    ::aie::vector<T, N> x1 = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> x2 = ::aie::load_v<N>(input + v + dims_half);
    ::aie::vector<T, 2 * N> cache = ::aie::load_v<2 * N>(lut + i);

    ::aie::vector<T, N> cos_val = ::aie::filter_even(cache, 1);
    ::aie::vector<T, N> sin_val = ::aie::filter_odd(cache, 1);

    // First half: x1*cos - x2*sin
    ::aie::vector<T, N> x1_cos = ::aie::mul(x1, cos_val);
    ::aie::vector<T, N> x2_sin = ::aie::mul(x2, sin_val);
    ::aie::vector<T, N> y_first_half = ::aie::sub(x1_cos, x2_sin);
    ::aie::store_v(output + v, y_first_half);

    // Second half: x2*cos + x1*sin
    ::aie::vector<T, N> x2_cos = ::aie::mul(x2, cos_val);
    ::aie::vector<T, N> x1_sin = ::aie::mul(x1, sin_val);
    ::aie::vector<T, N> y_second_half = ::aie::add(x2_cos, x1_sin);
    ::aie::store_v(output + v + dims_half, y_second_half);
  }
  event1();
}

extern "C" {
// Interleaved (Llama-paper) RoPE — the default; existing designs bind this.
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
  rope_kernel<bfloat16, 16>(input, lut, output, dims);
}

// Two-halves (HuggingFace-transformers) RoPE.
void rope_two_halves(bfloat16 *input, bfloat16 *lut, bfloat16 *output,
                     int32_t dims) {
  rope_kernel_two_halves<bfloat16, 32>(input, lut, output, dims);
}
}