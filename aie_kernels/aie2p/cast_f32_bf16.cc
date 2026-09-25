//===- cast_f32_bf16.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <cassert>
#include <stdint.h>

// Element-wise f32 -> bf16 narrowing cast over one row of `cols` elements.
//
// `aie::vector<float, N>` has no `to_vector<bfloat16>`
// directly, so the narrow goes through an `accfloat` accumulator, which does.
//
// The default AIE rounding is truncation toward zero; `conv_even` instead
// matches a host f32 -> bf16 pack (e.g. `_mm512_cvtne2ps_pbh` on AVX512-BF16),
// so an on-chip cast and its host equivalent agree bit-for-bit. The mode is
// one sticky register shared by every kernel on this core, so it is handed
// back before returning.
template <int N>
void cast_f32_bf16_row(const float *restrict input, bfloat16 *restrict output,
                       int32_t cols) {
  assert(cols % N == 0);
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
#if __AIE_ARCH__ == 20
  // AIE2 loads the f32 vector straight into the accumulator, which only the a
  // port can do, so a single chain pipelines to two cycles per 16 elements.
  // Its 5-stage schedule is only used for a loop known to run a few times.
  auto pin = ::aie::begin_restrict_vector<N>(input);
  auto pout = ::aie::begin_restrict_vector<N>(output);
  const int steps = (uint32_t)cols / N;
  if (steps >= 8) {
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < steps; i++) {
      ::aie::accum<accfloat, N> a;
      a.from_vector(*pin++);
      *pout++ = a.template to_vector<bfloat16>();
    }
  } else {
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < steps; i++) {
      ::aie::accum<accfloat, N> a;
      a.from_vector(*pin++);
      *pout++ = a.template to_vector<bfloat16>();
    }
  }
#else
  // Indexing off `i` costs a shift and a pointer update per iteration and
  // leaves a loop the pipeliner rejects ("the loop structure is not
  // supported"), so each 512-bit load stands alone with its latency exposed.
  // Walking the two pointers and unrolling by eight instead gets one load and
  // one converting store issued in the same bundle, eight elements' worth of
  // accumulators deep.
  const float *restrict in = input;
  bfloat16 *restrict out = output;
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  AIE_LOOP_UNROLL(8)
  for (int i = 0; i < cols; i += N, in += N, out += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(in);
    ::aie::accum<accfloat, N> a;
    a.from_vector(v);
    ::aie::store_v(out, a.template to_vector<bfloat16>());
  }
#endif
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void cast_f32_bf16_row(float *input, bfloat16 *output, int32_t cols) {
  cast_f32_bf16_row<16>(input, output, cols);
}
}
