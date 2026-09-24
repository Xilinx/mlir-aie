//===- swiglu.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include "activations.h" // tanh_bf16_v16
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SWIGLU_ELEMS
#define SWIGLU_ELEMS vector_size
#endif

// See add.cc: one bf16 vector register, 512 bits on AIE2P and 256 on AIE2.
#if __AIE_ARCH__ >= 21
#define SWIGLU_LANES 32
#else
#define SWIGLU_LANES 16
#endif

// out = (x * w1) * silu(x * w2), one vector register per iteration. tanh runs
// 16 lanes at a time whatever the register width, so the 32-wide iteration
// splits and re-concatenates and the 16-wide one does neither; silu.cc
// explains why that is a template parameter rather than an `if`.
//
// See sigmoid.cc for why 0.5 * (1 + tanh) is written as one mac against an
// accumulator preloaded with 0.5.
template <int lanes>
static inline void swiglu_impl(bfloat16 *restrict input_vector,
                               bfloat16 *restrict weight_vector_1,
                               bfloat16 *restrict weight_vector_2,
                               bfloat16 *restrict output_vector) {
  const int num_elems = SWIGLU_ELEMS;
  auto it_in = aie::begin_restrict_vector<lanes>((bfloat16 *)input_vector);
  auto it_wt_1 = aie::begin_restrict_vector<lanes>((bfloat16 *)weight_vector_1);
  auto it_wt_2 = aie::begin_restrict_vector<lanes>((bfloat16 *)weight_vector_2);
  auto it_out = aie::begin_restrict_vector<lanes>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, lanes> register_0_5_wide =
      aie::broadcast<bfloat16, lanes>(0.5f);
  aie::accum<accfloat, lanes> half;
  half.from_vector(register_0_5_wide);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < num_elems; i += lanes) {
    aie::vector<bfloat16, lanes> input = *it_in++;
    aie::vector<bfloat16, lanes> weight_1 = *it_wt_1++;
    aie::vector<bfloat16, lanes> weight_2 = *it_wt_2++;

    aie::vector<bfloat16, lanes> mul_input_weight_1 = aie::mul(input, weight_1);
    aie::vector<bfloat16, lanes> mul_input_weight_2 = aie::mul(input, weight_2);

    aie::vector<bfloat16, lanes> tanh_half_x;
    if constexpr (lanes == 32) {
      auto lo = tanh_bf16_v16(
          aie::mul(mul_input_weight_2.template extract<16>(0), register_0_5));
      auto hi = tanh_bf16_v16(
          aie::mul(mul_input_weight_2.template extract<16>(1), register_0_5));
      tanh_half_x = aie::concat(lo, hi);
    } else {
      tanh_half_x = tanh_bf16_v16(aie::mul(mul_input_weight_2, register_0_5));
    }

    aie::vector<bfloat16, lanes> sigmoid_approx =
        aie::mac(half, tanh_half_x, register_0_5_wide)
            .template to_vector<bfloat16>();
    aie::vector<bfloat16, lanes> silu_output =
        aie::mul(mul_input_weight_2, sigmoid_approx);

    auto mul_output = aie::mul(mul_input_weight_1, silu_output);

    *it_out++ = mul_output.template to_vector<bfloat16>();
  }
}

void swiglu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                             bfloat16 *restrict weight_vector_1,
                             bfloat16 *restrict weight_vector_2,
                             bfloat16 *restrict output_vector,
                             const int32_t vector_size) {
  event0();
  swiglu_impl<SWIGLU_LANES>(input_vector, weight_vector_1, weight_vector_2,
                            output_vector);
  event1();

  return;
}

extern "C" {

void swiglu_bf16(bfloat16 *restrict input, bfloat16 *restrict weights_1,
                 bfloat16 *restrict weights_2, bfloat16 *restrict output) {
  // Assuming input size is a multiple of SWIGLU_LANES
  int32_t input_size = 1024;
  swiglu_tanh_approx_bf16(input, weights_1, weights_2, output, input_size);
}

} // extern "C"
