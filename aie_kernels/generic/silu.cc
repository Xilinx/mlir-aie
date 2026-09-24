//===- silu.cc --------------------------------------------*- C++
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

#ifndef SILU_ELEMS
#define SILU_ELEMS vector_size
#endif

// See add.cc: one bf16 vector register, 512 bits on AIE2P and 256 on AIE2.
#if __AIE_ARCH__ >= 21
#define SILU_LANES 32
#else
#define SILU_LANES 16
#endif

// silu(x) = x * 0.5 * (1 + tanh(x/2)), one vector register per iteration.
// tanh runs 16 lanes at a time whatever the register width, so the 32-wide
// iteration splits and re-concatenates and the 16-wide one does neither.
// Templated on the lane count so `if constexpr` genuinely discards the branch
// this architecture does not take. In a plain function both branches still
// have to be well-formed, and the 16-lane one is not at 32 lanes.
//
// See sigmoid.cc for why 0.5 * (1 + tanh) is written as one mac against an
// accumulator preloaded with 0.5. What is left is still a chain the target
// cannot fill from a single iteration -- the AIE2P body schedules at II19
// with seven of its bundles empty -- so four iterations are unrolled into it,
// which packs the same II with four times the work.
//
// Only where tanh is the single vtanh, though. The LUT path's body is already
// around eighty bundles and is limited by its table loads rather than by an
// unfilled schedule, so unrolling it buys 13% and costs 848 bytes of program
// memory (II77 over 32 iterations at 576 bytes, against II269 over 8 at 1424).
template <int lanes>
static inline void silu_impl(bfloat16 *restrict input_vector,
                             bfloat16 *restrict output_vector) {
  const int num_elems = SILU_ELEMS;
  auto it_in = aie::begin_restrict_vector<lanes>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<lanes>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, lanes> register_0_5_wide =
      aie::broadcast<bfloat16, lanes>(0.5f);
  aie::accum<accfloat, lanes> half;
  half.from_vector(register_0_5_wide);
  AIE_PREPARE_FOR_PIPELINING
#if ACTIVATIONS_NATIVE_TANH
  AIE_LOOP_UNROLL(4)
#endif
  for (int i = 0; i < num_elems; i += lanes) {
    auto input = *it_in++;

    aie::vector<bfloat16, lanes> tanh_half_x;
    if constexpr (lanes == 32) {
      auto lo =
          tanh_bf16_v16(aie::mul(input.template extract<16>(0), register_0_5));
      auto hi =
          tanh_bf16_v16(aie::mul(input.template extract<16>(1), register_0_5));
      tanh_half_x = aie::concat(lo, hi);
    } else {
      tanh_half_x = tanh_bf16_v16(aie::mul(input, register_0_5));
    }

    aie::vector<bfloat16, lanes> sigmoid_approx =
        aie::mac(half, tanh_half_x, register_0_5_wide)
            .template to_vector<bfloat16>();
    auto mul_output = aie::mul(input, sigmoid_approx);

    *it_out++ = mul_output.template to_vector<bfloat16>();
  }
}

void silu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                           bfloat16 *restrict output_vector,
                           const int32_t vector_size) {
  event0();
  silu_impl<SILU_LANES>(input_vector, output_vector);
  event1();

  return;
}

extern "C" {

void silu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  int32_t input_size = 1024; // Assuming input size is a multiple of 32
  silu_tanh_approx_bf16(input, output, input_size);
}

void silu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  silu_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
