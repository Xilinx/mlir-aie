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

#if AIE_TUNED_AIE2P && !ACTIVATIONS_NATIVE_TANH
#include "aie_bank_placement.h"
// AIE2P's LUT sigmoid reads its own table rather than tanh's: tanh_lut_ab/cd
// with each segment rewritten for 0.5 + 0.5 * tanh(x/2), the slope over 4 and
// the offset 0.5 + 0.5 * offset, indexed by x in segments of 0.5 over [-8, 8).
// That drops the x/2 and 0.5 * (1 + t) passes around the table reads and one
// of the two bf16 roundings. Every slope is still a bf16, which the lookup
// reads as the high half of each float.
// clang-format off
AIE_BANK_A alignas(aie::vector_decl_align) float sigmoid_lut_ab[128] = {
    0.0f, 0.0f,
    0.00070953369140625f, 0.005859375f,
    0.00127410888671875f, 0.009765625f,
    0.0018768310546875f, 0.013671875f,
    0.0f, 0.0f,
    0.00070953369140625f, 0.005859375f,
    0.00127410888671875f, 0.009765625f,
    0.0018768310546875f, 0.013671875f,
    0.003173828125f, 0.021484375f,
    0.00531005859375f, 0.033203125f,
    0.00885009765625f, 0.05078125f,
    0.01409912109375f, 0.07421875f,
    0.003173828125f, 0.021484375f,
    0.00531005859375f, 0.033203125f,
    0.00885009765625f, 0.05078125f,
    0.01409912109375f, 0.07421875f,
    0.02294921875f, 0.109375f,
    0.036376953125f, 0.15625f,
    0.057373046875f, 0.21875f,
    0.0869140625f, 0.2919921875f,
    0.02294921875f, 0.109375f,
    0.036376953125f, 0.15625f,
    0.057373046875f, 0.21875f,
    0.0869140625f, 0.2919921875f,
    0.1259765625f, 0.3701171875f,
    0.1728515625f, 0.440185546875f,
    0.216796875f, 0.484619140625f,
    0.25f, 0.5f,
    0.1259765625f, 0.3701171875f,
    0.1728515625f, 0.440185546875f,
    0.216796875f, 0.484619140625f,
    0.25f, 0.5f,
    0.25f, 0.5f,
    0.216796875f, 0.515380859375f,
    0.1728515625f, 0.559814453125f,
    0.1259765625f, 0.6298828125f,
    0.25f, 0.5f,
    0.216796875f, 0.515380859375f,
    0.1728515625f, 0.559814453125f,
    0.1259765625f, 0.6298828125f,
    0.0869140625f, 0.7080078125f,
    0.057373046875f, 0.78125f,
    0.036376953125f, 0.84375f,
    0.02294921875f, 0.890625f,
    0.0869140625f, 0.7080078125f,
    0.057373046875f, 0.78125f,
    0.036376953125f, 0.84375f,
    0.02294921875f, 0.890625f,
    0.01409912109375f, 0.92578125f,
    0.00885009765625f, 0.94921875f,
    0.00531005859375f, 0.966796875f,
    0.003173828125f, 0.978515625f,
    0.01409912109375f, 0.92578125f,
    0.00885009765625f, 0.94921875f,
    0.00531005859375f, 0.966796875f,
    0.003173828125f, 0.978515625f,
    0.0018768310546875f, 0.986328125f,
    0.00127410888671875f, 0.990234375f,
    0.00070953369140625f, 0.994140625f,
    0.0f, 1.0f,
    0.0018768310546875f, 0.986328125f,
    0.00127410888671875f, 0.990234375f,
    0.00070953369140625f, 0.994140625f,
    0.0f, 1.0f};
AIE_BANK_B alignas(aie::vector_decl_align) float sigmoid_lut_cd[128] = {
    0.0f, 0.0f,
    0.00070953369140625f, 0.005859375f,
    0.00127410888671875f, 0.009765625f,
    0.0018768310546875f, 0.013671875f,
    0.0f, 0.0f,
    0.00070953369140625f, 0.005859375f,
    0.00127410888671875f, 0.009765625f,
    0.0018768310546875f, 0.013671875f,
    0.003173828125f, 0.021484375f,
    0.00531005859375f, 0.033203125f,
    0.00885009765625f, 0.05078125f,
    0.01409912109375f, 0.07421875f,
    0.003173828125f, 0.021484375f,
    0.00531005859375f, 0.033203125f,
    0.00885009765625f, 0.05078125f,
    0.01409912109375f, 0.07421875f,
    0.02294921875f, 0.109375f,
    0.036376953125f, 0.15625f,
    0.057373046875f, 0.21875f,
    0.0869140625f, 0.2919921875f,
    0.02294921875f, 0.109375f,
    0.036376953125f, 0.15625f,
    0.057373046875f, 0.21875f,
    0.0869140625f, 0.2919921875f,
    0.1259765625f, 0.3701171875f,
    0.1728515625f, 0.440185546875f,
    0.216796875f, 0.484619140625f,
    0.25f, 0.5f,
    0.1259765625f, 0.3701171875f,
    0.1728515625f, 0.440185546875f,
    0.216796875f, 0.484619140625f,
    0.25f, 0.5f,
    0.25f, 0.5f,
    0.216796875f, 0.515380859375f,
    0.1728515625f, 0.559814453125f,
    0.1259765625f, 0.6298828125f,
    0.25f, 0.5f,
    0.216796875f, 0.515380859375f,
    0.1728515625f, 0.559814453125f,
    0.1259765625f, 0.6298828125f,
    0.0869140625f, 0.7080078125f,
    0.057373046875f, 0.78125f,
    0.036376953125f, 0.84375f,
    0.02294921875f, 0.890625f,
    0.0869140625f, 0.7080078125f,
    0.057373046875f, 0.78125f,
    0.036376953125f, 0.84375f,
    0.02294921875f, 0.890625f,
    0.01409912109375f, 0.92578125f,
    0.00885009765625f, 0.94921875f,
    0.00531005859375f, 0.966796875f,
    0.003173828125f, 0.978515625f,
    0.01409912109375f, 0.92578125f,
    0.00885009765625f, 0.94921875f,
    0.00531005859375f, 0.966796875f,
    0.003173828125f, 0.978515625f,
    0.0018768310546875f, 0.986328125f,
    0.00127410888671875f, 0.990234375f,
    0.00070953369140625f, 0.994140625f,
    0.0f, 1.0f,
    0.0018768310546875f, 0.986328125f,
    0.00127410888671875f, 0.990234375f,
    0.00070953369140625f, 0.994140625f,
    0.0f, 1.0f};
// clang-format on

__attribute__((always_inline)) inline aie::vector<bfloat16, 16>
sigmoid_lut_bf16(aie::vector<bfloat16, 16> x) {
  return lut_segments_acc<5>(sigmoid_lut_ab, sigmoid_lut_cd, x)
      .to_vector<bfloat16>();
}
#endif

#ifndef SIGMOID_ELEMS
#define SIGMOID_ELEMS vector_size
#endif

// sigmoid(x) = 0.5 * (1 + tanh(x/2)), 32 bf16 elements per iteration, with
// tanh(x/2) on the two 16-lane halves both tanh paths work in. Passing the
// multiply's accumulator straight in keeps x/2 in f32 on AIE2P; AIE2's LUT
// narrows it, which is the accuracy difference between the two architectures.
// AIE2P's LUT build reads the table above instead.
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
  // The loop of tanh_lut_map, whose comment has why 16 is asked for.
  auto it_in = aie::begin_vector<32>(input_vector);
  auto it_out = aie::begin_vector<32>(output_vector);
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < num_elems; i += 32) {
    const aie::vector<bfloat16, 32> x = *it_in++;
    *it_out++ = aie::concat(sigmoid_lut_bf16(x.extract<16>(0)),
                            sigmoid_lut_bf16(x.extract<16>(1)));
  }
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
