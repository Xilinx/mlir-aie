//===- sigmoid.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h" // tanh_bf16_v16
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SIGMOID_ELEMS
#define SIGMOID_ELEMS vector_size
#endif

// sigmoid(x) = 0.5 * (1 + tanh(x/2)), 32 bf16 elements per iteration, with
// tanh(x/2) on the two 16-lane halves both tanh paths work in. Passing the
// multiply's accumulator straight in keeps x/2 in f32 on AIE2P; AIE2's LUT
// narrows it, which is the accuracy difference between the two architectures.
//
// 0.5 * (1 + t) is one mac, t * 0.5 onto an accumulator holding 0.5, rather
// than an add and a multiply. Scaling by 0.5 is exact, so the result is the
// same.
void sigmoid_tanh_approx_bf16(bfloat16 *restrict input_vector,
                              bfloat16 *restrict output_vector,
                              const int32_t vector_size) {
  event0();

  const int num_elems = SIGMOID_ELEMS;
#if AIE_TUNED_AIE2
  // AIE2's tanh reads a table; lut_map_bf16 lays the loop out around the reads.
  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> half;
  half.from_vector(register_0_5);
  lut_map_bf16<4, false>(
      input_vector, output_vector, num_elems, [&](aie::vector<bfloat16, 16> x) {
        return aie::vector<bfloat16, 16>(
            aie::mac(half, tanh_bf16_v16(aie::mul(x, register_0_5)),
                     register_0_5)
                .to_vector<bfloat16>());
      });
#elif AIE_TUNED_AIE2P && !ACTIVATIONS_NATIVE_TANH
  // The LUT tanh pipelines only alone in its loop, so x/2 goes to the output
  // first, tanh_lut_map rewrites it, and a third pass makes 0.5 * (1 + t).
  // The LUT narrows x/2 to bf16 either way, so the result is the same.
  aie::vector<bfloat16, 32> register_0_5_wide =
      aie::broadcast<bfloat16, 32>(0.5f);
  auto it_in = aie::begin_restrict_vector<32>(input_vector);
  auto it_half_x = aie::begin_restrict_vector<32>(output_vector);
  for (int i = 0; i < num_elems; i += 32)
    *it_half_x++ = aie::mul(*it_in++, register_0_5_wide).to_vector<bfloat16>();

  tanh_lut_map(output_vector, output_vector, num_elems);

  aie::accum<accfloat, 32> half;
  half.from_vector(register_0_5_wide);
  auto it_tanh = aie::begin_vector<32>(output_vector);
  auto it_out = aie::begin_vector<32>(output_vector);
  for (int i = 0; i < num_elems; i += 32)
    *it_out++ =
        aie::mac(half, *it_tanh++, register_0_5_wide).to_vector<bfloat16>();
#else
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 32> register_0_5_wide =
      aie::broadcast<bfloat16, 32>(0.5f);
  aie::accum<accfloat, 32> half;
  half.from_vector(register_0_5_wide);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < num_elems; i += 32) {
    auto input = *it_in++;

    auto tanh_lo = tanh_bf16_v16(aie::mul(input.extract<16>(0), register_0_5));
    auto tanh_hi = tanh_bf16_v16(aie::mul(input.extract<16>(1), register_0_5));
    aie::vector<bfloat16, 32> tanh_half_x = aie::concat(tanh_lo, tanh_hi);

    *it_out++ =
        aie::mac(half, tanh_half_x, register_0_5_wide).to_vector<bfloat16>();
  }
#endif

  event1();

  return;
}

extern "C" {

void sigmoid_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  int input_size) {
  sigmoid_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
