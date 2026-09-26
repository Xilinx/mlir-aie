//===- silu.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h" // tanh_bf16_v16
#include "sigmoid_lut.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SILU_ELEMS
#define SILU_ELEMS vector_size
#endif

// silu(x) = x * 0.5 * (1 + tanh(x/2)), one vector register per iteration.
// tanh runs 16 lanes at a time whatever the register width, so the 32-wide
// iteration splits and re-concatenates and the 16-wide one does neither.
// Templated on the lane count so `if constexpr` genuinely discards the branch
// this architecture does not take. In a plain function both branches still
// have to be well-formed, and the 16-lane one is not at 32 lanes.
//
// See sigmoid.cc for the 0.5 * (1 + tanh) mac. The unroll fills the native
// vtanh's latency chain; the LUT tanh is bound by its table loads instead.
// x is clamped at -8 before it multiplies the sigmoid, as in silu_aie2. On
// AIE2P the clamp in the tanh loop takes it from II 24 to 59 per four
// vectors, so the sigmoid goes to the output and a second pass at II 1
// multiplies it by the clamped x.
template <int lanes>
static inline void silu_impl(bfloat16 *restrict input_vector,
                             bfloat16 *restrict output_vector,
                             const int32_t vector_size) {
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
#if AIE_TUNED_AIE2P
    *it_out++ = sigmoid_approx;
#else
    auto mul_output =
        aie::mul(aie::max(input, bfloat16(-8.0f)), sigmoid_approx);

    *it_out++ = mul_output.template to_vector<bfloat16>();
#endif
  }
#if AIE_TUNED_AIE2P
  auto it_x = aie::begin_restrict_vector<lanes>((bfloat16 *)input_vector);
  auto it_sig = aie::begin_vector<lanes>((bfloat16 *)output_vector);
  auto it_y = aie::begin_vector<lanes>((bfloat16 *)output_vector);
#pragma clang loop pipeline_initiation_interval(1)
  for (int i = 0; i < num_elems; i += lanes)
    *it_y++ = aie::mul(aie::max(*it_x++, bfloat16(-8.0f)), *it_sig++)
                  .template to_vector<bfloat16>();
#endif
}

#if AIE_TUNED_AIE2
// AIE2's tanh reads a table; lut_map_bf16 lays the loop out around the reads.
// The sigmoid is exactly 0 from x = -8 down, so x is clamped there before it
// multiplies it, which keeps -inf from making -inf * 0.
static inline void silu_aie2(bfloat16 *restrict input_vector,
                             bfloat16 *restrict output_vector,
                             const int32_t vector_size) {
  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> half;
  half.from_vector(register_0_5);
  lut_map_bf16(input_vector, output_vector, SILU_ELEMS,
               [&](aie::vector<bfloat16, 16> x) {
                 aie::vector<bfloat16, 16> sigmoid_approx =
                     aie::mac(half, tanh_bf16_v16(aie::mul(x, register_0_5)),
                              register_0_5)
                         .to_vector<bfloat16>();
                 return aie::vector<bfloat16, 16>(
                     aie::mul(aie::max(x, bfloat16(-8.0f)), sigmoid_approx)
                         .to_vector<bfloat16>());
               });
}
#endif

#if AIE_TUNED_AIE2P && !ACTIVATIONS_NATIVE_TANH
// sigmoid.cc's table to the output, then x times it, clamped at -8 as in
// silu_aie2, in a second pass at II 1.
static inline void silu_lut_aie2p(bfloat16 *restrict input_vector,
                                  bfloat16 *restrict output_vector,
                                  const int32_t vector_size) {
  const int num_elems = SILU_ELEMS;
  auto it_in = aie::begin_restrict_vector<32>(input_vector);
  auto it_sig = aie::begin_restrict_vector<32>(output_vector);
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < num_elems; i += 32) {
    const aie::vector<bfloat16, 32> x = *it_in++;
    *it_sig++ = aie::concat(sigmoid_lut_bf16(x.extract<16>(0)),
                            sigmoid_lut_bf16(x.extract<16>(1)));
  }

  auto it_x = aie::begin_restrict_vector<32>(input_vector);
  auto it_s = aie::begin_vector<32>(output_vector);
  auto it_out = aie::begin_vector<32>(output_vector);
#pragma clang loop pipeline_initiation_interval(1)
  for (int i = 0; i < num_elems; i += 32)
    *it_out++ = aie::mul(aie::max(*it_x++, bfloat16(-8.0f)), *it_s++)
                    .to_vector<bfloat16>();
}
#endif

void silu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                           bfloat16 *restrict output_vector,
                           const int32_t vector_size) {
  event0();
#if AIE_TUNED_AIE2
  silu_aie2(input_vector, output_vector, vector_size);
#elif AIE_TUNED_AIE2P && !ACTIVATIONS_NATIVE_TANH
  silu_lut_aie2p(input_vector, output_vector, vector_size);
#else
  silu_impl<AIE_BF16_LANES>(input_vector, output_vector, vector_size);
#endif
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
