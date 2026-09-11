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

    // Each half stays in the accumulator until one rounding at the end.
    // Rounding the products to bfloat16 first spends the result's significant
    // bits on digits that then cancel: a rotation subtracts two products of
    // similar size, and a pair cancelling to ~1e-4 from operands of order 1
    // keeps almost none of them.
    ::aie::vector<T, N / 2> output_even =
        ::aie::msc(::aie::mul(x_even, cos_val), x_odd, sin_val)
            .template to_vector<T>();
    ::aie::vector<T, N / 2> output_odd =
        ::aie::mac(::aie::mul(x_even, sin_val), x_odd, cos_val)
            .template to_vector<T>();

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

    // First half: x1*cos - x2*sin, accumulated then rounded once (see
    // rope_kernel above for why the intermediate products must not round).
    ::aie::vector<T, N> y_first_half =
        ::aie::msc(::aie::mul(x1, cos_val), x2, sin_val)
            .template to_vector<T>();
    ::aie::store_v(output + v, y_first_half);

    // Second half: x2*cos + x1*sin
    ::aie::vector<T, N> y_second_half =
        ::aie::mac(::aie::mul(x2, cos_val), x1, sin_val)
            .template to_vector<T>();
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
