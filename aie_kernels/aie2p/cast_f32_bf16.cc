//===- cast_f32_bf16.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    ::aie::accum<accfloat, N> a;
    a.from_vector(v);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void cast_f32_bf16_row(float *input, bfloat16 *output, int32_t cols) {
  cast_f32_bf16_row<16>(input, output, cols);
}
}
