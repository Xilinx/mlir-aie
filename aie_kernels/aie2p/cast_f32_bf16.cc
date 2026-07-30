//===- cast_f32_bf16.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

// Element-wise f32 -> bf16 narrowing cast, round-to-nearest-even.
//
// A resident on-chip dataflow that hands an intermediate activation from an
// f32-producing kernel (e.g. a numerically-stable norm or a K-split f32
// accumulate) to a bf16-consuming one (e.g. a bf16 matmul) needs this cast to
// stay on-chip; otherwise the f32 buffer has to round-trip through the host to
// be requantized. `aie::vector<float, N>` has no `to_vector<bfloat16>`
// directly, so the narrow goes through an `accfloat` accumulator, which does.
//
// One call processes one row of `cols` elements; `cols` must be a multiple of
// N (16). Rounding is round-to-nearest-even (`conv_even`) rather than the AIE
// default (truncation toward zero), matching a host f32->bf16 pack that also
// rounds to nearest even (e.g. `_mm512_cvtne2ps_pbh` on AVX512-BF16) so an
// on-chip cast and its host equivalent agree bit-for-bit.
//
// The rounding mode is a single sticky register shared by every kernel that
// runs on this core, so this kernel saves the caller's mode and restores it
// before returning rather than leaving conv_even set for whatever runs next,
// the same save/restore mm.cc's AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
// block uses.
template <int N>
void cast_f32_bf16_row(const float *restrict input, bfloat16 *restrict output,
                       int32_t cols) {
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
