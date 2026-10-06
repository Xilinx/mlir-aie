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

// One interleaved-RoPE step over N elements.
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
  // A row runs in 4N-element steps and closes with at most three N-element
  // ones. Both divisors are unsigned so they stay shifts.
  constexpr int W = 4 * N;
  const int wide = (uint32_t)dims / W;
  const int tail = ((uint32_t)dims % W) / N;
  // Walking cursors rather than three addresses recomputed from one index.
  const T *restrict pi = input;
  const T *restrict pl = lut;
  T *restrict po = output;
  // The pipelined schedule needs a promised trip count; a short row runs the
  // same steps unpipelined.
  if (AIE_TUNED_AIE2P && wide >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int k = 0; k < wide; ++k, pi += W, pl += W, po += W) {
      rope_step<T, W>(pi, pl, po);
    }
  } else if (wide > 0) {
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

// One two-halves step over N elements of each half. Aligned says the second
// half starts on a whole vector.
template <typename T, int N, bool Aligned = false>
static inline void rope_halves_step(const T *restrict input,
                                    const T *restrict lut, T *restrict output,
                                    int dims_half) {
  ::aie::vector<T, N> x1 = ::aie::load_v<N>(input);
  // For dims = 96, the second half is only 32-byte aligned, not the
  // 64-byte alignment required by AIE2P's 32-lane bf16 loads/stores.
  ::aie::vector<T, N> x2 = Aligned
                               ? ::aie::load_v<N>(input + dims_half)
                               : ::aie::load_unaligned_v<N>(input + dims_half);
  ::aie::vector<T, 2 * N> cache = ::aie::load_v<2 * N>(lut);

  ::aie::vector<T, N> cos_val = ::aie::filter_even(cache, 1);
  ::aie::vector<T, N> sin_val = ::aie::filter_odd(cache, 1);

  // First half: x1*cos - x2*sin, accumulated then rounded once (see
  // rope_step above for why the intermediate products must not round).
  ::aie::store_v(
      output,
      ::aie::msc(::aie::mul(x1, cos_val), x2, sin_val).template to_vector<T>());
  // Second half: x2*cos + x1*sin
  ::aie::vector<T, N> y2 =
      ::aie::mac(::aie::mul(x2, cos_val), x1, sin_val).template to_vector<T>();
  if constexpr (Aligned)
    ::aie::store_v(output + dims_half, y2);
  else
    ::aie::store_unaligned_v(output + dims_half, y2);
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
  // Walking cursors, as in rope_kernel.
  const T *restrict pi = input;
  const T *restrict pl = lut;
  T *restrict po = output;
#if AIE_TUNED_AIE2P
  // A half of whole vectors needs no unaligned read-modify-write; the loop
  // pipelines with a promised trip count, as in rope_kernel.
  if ((uint32_t)dims_half % N == 0) {
    if (wide >= 4) {
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int k = 0; k < wide; ++k, pi += N, pl += 2 * N, po += N) {
        rope_halves_step<T, N, true>(pi, pl, po, dims_half);
      }
    } else {
      for (int k = 0; k < wide; ++k, pi += N, pl += 2 * N, po += N) {
        rope_halves_step<T, N, true>(pi, pl, po, dims_half);
      }
    }
    event1();
    return;
  }
#endif
  if (wide > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int k = 0; k < wide; ++k, pi += N, pl += 2 * N, po += N) {
      rope_halves_step<T, N>(pi, pl, po, dims_half);
    }
  }
  // IRON only accepts a two-halves row that is a multiple of 2N, so each half
  // is a multiple of N/2 and what is left here is one half-width step or
  // nothing.
  if (wide * N < dims_half) {
    rope_halves_step<T, N / 2>(pi, pl, po, dims_half);
  }
  event1();
}

#if AIE_TUNED_AIE2
// AIE2 multiplies bf16 16 lanes at a time, so aie::mul above pads each operand
// with zeros. A bf16 mac instead sums two products into each f32 lane, lane i
// getting a[i] b[i] + a[i + 16] b[i + 16], so each rotated half is one mac:
// [x | y] [cos | -sin] and [y | x] [cos | sin].
static inline v32bfloat16 negate_high(v32bfloat16 v) {
  const v32uint16 sign_high = ::aie::concat(
      ::aie::zeros<uint16_t, 16>(), ::aie::broadcast<uint16_t, 16>(0x8000));
  return __builtin_bit_cast(v32bfloat16,
                            __builtin_bit_cast(v32uint16, v) + sign_high);
}

static inline v32bfloat16 swap_halves(v32bfloat16 v) {
  return concat(extract_v16bfloat16(v, 1), extract_v16bfloat16(v, 0));
}

// [cos | sin] for 16 rotations.
static inline v32bfloat16 load_cos_sin(const bfloat16 *restrict lut) {
  return shuffle(v32bfloat16(::aie::load_v<32>(lut)), T16_16x2);
}

// One interleaved step over 32 elements.
static inline void rope_step_aie2(const bfloat16 *restrict input,
                                  const bfloat16 *restrict lut,
                                  bfloat16 *restrict output) {
  v32bfloat16 x = shuffle(v32bfloat16(::aie::load_v<32>(input)), T16_16x2);
  v32bfloat16 cs = load_cos_sin(lut);
  v32bfloat16 y = concat(to_v16bfloat16(mul_elem_16_2(x, negate_high(cs))),
                         to_v16bfloat16(mul_elem_16_2(swap_halves(x), cs)));
  ::aie::store_v(output, ::aie::vector<bfloat16, 32>(shuffle(y, T16_2x16)));
}

static void rope_aie2(const bfloat16 *restrict input,
                      const bfloat16 *restrict lut, bfloat16 *restrict output,
                      int32_t dims) {
  event0();
  constexpr unsigned MIN_STEPS = 4;
  const unsigned steps = (uint32_t)dims / 32;
  const bfloat16 *restrict pi = input;
  const bfloat16 *restrict pl = lut;
  bfloat16 *restrict po = output;
  if (steps >= MIN_STEPS) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_STEPS)
    for (unsigned k = 0; k < steps; ++k, pi += 32, pl += 32, po += 32)
      rope_step_aie2(pi, pl, po);
  } else {
    for (unsigned k = 0; k < steps; ++k, pi += 32, pl += 32, po += 32)
      rope_step_aie2(pi, pl, po);
  }
  if ((uint32_t)dims & 16)
    rope_step<bfloat16, 16>(pi, pl, po);
  event1();
}

// One two-halves step over 16 elements of each half.
static inline void rope_halves_step_aie2(const bfloat16 *restrict input,
                                         const bfloat16 *restrict lut,
                                         bfloat16 *restrict output,
                                         unsigned dims_half) {
  v16bfloat16 x1 = ::aie::load_v<16>(input);
  v16bfloat16 x2 = ::aie::load_v<16>(input + dims_half);
  v32bfloat16 cs = load_cos_sin(lut);
  ::aie::store_v(output, ::aie::vector<bfloat16, 16>(to_v16bfloat16(
                             mul_elem_16_2(concat(x1, x2), negate_high(cs)))));
  ::aie::store_v(output + dims_half, ::aie::vector<bfloat16, 16>(to_v16bfloat16(
                                         mul_elem_16_2(concat(x2, x1), cs))));
}

// A two-halves row is a multiple of 32, so each half is whole 16-element steps.
static void rope_two_halves_aie2(const bfloat16 *restrict input,
                                 const bfloat16 *restrict lut,
                                 bfloat16 *restrict output, int32_t dims) {
  event0();
  constexpr unsigned MIN_STEPS = 4;
  const unsigned dims_half = (uint32_t)dims / 2;
  const unsigned steps = dims_half / 16;
  const bfloat16 *restrict pi = input;
  const bfloat16 *restrict pl = lut;
  bfloat16 *restrict po = output;
  if (steps >= MIN_STEPS) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_STEPS)
    for (unsigned k = 0; k < steps; ++k, pi += 16, pl += 32, po += 16)
      rope_halves_step_aie2(pi, pl, po, dims_half);
  } else {
    for (unsigned k = 0; k < steps; ++k, pi += 16, pl += 32, po += 16)
      rope_halves_step_aie2(pi, pl, po, dims_half);
  }
  event1();
}
#endif

extern "C" {
// Interleaved (Llama-paper) RoPE — the default; existing designs bind this.
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
#if AIE_TUNED_AIE2
  rope_aie2(input, lut, output, dims);
#else
  rope_kernel<bfloat16, 16>(input, lut, output, dims);
#endif
}

// Two-halves (HuggingFace-transformers) RoPE.
void rope_two_halves(bfloat16 *input, bfloat16 *lut, bfloat16 *output,
                     int32_t dims) {
#if AIE_TUNED_AIE2
  rope_two_halves_aie2(input, lut, output, dims);
#else
  rope_kernel_two_halves<bfloat16, 32>(input, lut, output, dims);
#endif
}
}
