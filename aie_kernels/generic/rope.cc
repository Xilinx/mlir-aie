//===- rope.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// One interleaved-RoPE step over N elements.  The shuffle/multiply/round chain
// below is the same instruction sequence at every N a vector register can
// hold, so the caller runs it at the widest N a row has room for.
template <typename T, int N>
static inline void rope_step(const T *restrict input, const T *restrict lut,
                             T *restrict output) {
  ::aie::vector<T, N> x = ::aie::load_v<N>(input);
  ::aie::vector<T, N> cache = ::aie::load_v<N>(lut);

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
  ::aie::store_v(output, ::aie::concat(low, high));
}

template <typename T, int N>
void rope_kernel(const T *restrict input, const T *restrict lut,
                 T *restrict output, int32_t dims) {
  event0();
  // A 4N-element step issues one vshuffle, one vmul and one vconv per operand
  // just as an N-element one does -- the narrow form leaves three quarters of
  // each 512-bit lane idle -- so a row runs in 4N-element steps and closes with
  // at most three N-element ones.  Both divisors are unsigned so they stay
  // shifts rather than becoming the 64-bit magic multiply.
  constexpr int W = 4 * N;
  const int wide = (uint32_t)dims / W;
  const int tail = ((uint32_t)dims % W) / N;
  // Walking cursors: recomputing input + v, lut + v and output + v from one
  // index gives the scheduler three address chains that all depend on v, and
  // it then schedules the step at II 30 rather than II 16.
  const T *restrict pi = input;
  const T *restrict pl = lut;
  T *restrict po = output;
  if (wide > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int k = 0; k < wide; ++k, pi += W, pl += W, po += W) {
      rope_step<T, W>(pi, pl, po);
    }
  }
  if (tail > 0) {
    AIE_LOOP_RANGE(1, 3)
    for (int k = 0; k < tail; ++k, pi += N, pl += N, po += N) {
      rope_step<T, N>(pi, pl, po);
    }
  }
  event1();
}

// One two-halves step over N elements of each half.
template <typename T, int N>
static inline void rope_halves_step(const T *restrict input,
                                    const T *restrict lut, T *restrict output,
                                    int dims_half) {
  ::aie::vector<T, N> x1 = ::aie::load_v<N>(input);
  // For dims = 96, the second half is only 32-byte aligned, not the
  // 64-byte alignment required by AIE2P's 32-lane bf16 loads/stores.
  ::aie::vector<T, N> x2 = ::aie::load_unaligned_v<N>(input + dims_half);
  ::aie::vector<T, 2 * N> cache = ::aie::load_v<2 * N>(lut);

  ::aie::vector<T, N> cos_val = ::aie::filter_even(cache, 1);
  ::aie::vector<T, N> sin_val = ::aie::filter_odd(cache, 1);

  // First half: x1*cos - x2*sin, accumulated then rounded once (see
  // rope_step above for why the intermediate products must not round).
  ::aie::store_v(
      output,
      ::aie::msc(::aie::mul(x1, cos_val), x2, sin_val).template to_vector<T>());
  // Second half: x2*cos + x1*sin
  ::aie::store_unaligned_v(
      output + dims_half,
      ::aie::mac(::aie::mul(x2, cos_val), x1, sin_val).template to_vector<T>());
}

// Two-halves RoPE (the layout used by HuggingFace transformers): the first and
// second halves of the vector are rotated against each other, rather than the
// even/odd interleave of the Llama-paper method in rope_kernel above.  Ported
// from IRON so designs targeting HF-style weights have a matching kernel.
template <typename T, int N>
void rope_kernel_two_halves(const T *restrict input, const T *restrict lut,
                            T *restrict output, int32_t dims) {
  event0();
  const int dims_half = (uint32_t)dims / 2;
  const int wide = (uint32_t)dims_half / N;
  // Walking cursors, for the reason given in rope_kernel above: from an index
  // the step schedules at II 45, from cursors at II 19.
  const T *restrict pi = input;
  const T *restrict pl = lut;
  T *restrict po = output;
  if (wide > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int k = 0; k < wide; ++k, pi += N, pl += 2 * N, po += N) {
      rope_halves_step<T, N>(pi, pl, po, dims_half);
    }
  }
  // IRON only accepts a two-halves row that is a multiple of 2N, so each half
  // is a multiple of N/2 and what is left here is one half-width step or
  // nothing.  A scalar close-out would instead pay per element for a float
  // multiply that aie2p has no scalar instruction for.
  if (wide * N < dims_half) {
    rope_halves_step<T, N / 2>(pi, pl, po, dims_half);
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
