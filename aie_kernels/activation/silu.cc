//===- silu.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SILU_ELEMS
#define SILU_ELEMS n
#endif

// x sigmoid(x) of bf16 from sigmoid_f32, its factor x held at +-184, so -inf
// gives -0; past 128, where sigmoid is 1, and for NaN, x passes through.
static inline void silu_f32(const bfloat16 *restrict x, bfloat16 *restrict y,
                            int32_t n) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec top = aie::broadcast<bfloat16, f32_lanes>(184.0f);
  const bf16_vec bottom = aie::broadcast<bfloat16, f32_lanes>(-184.0f);
  const bf16_vec zero = aie::zeros<bfloat16, f32_lanes>();
  const auto top_bits = aie::broadcast<int16_t, f32_lanes>(0x4300);
  const auto magnitude = aie::broadcast<uint16_t, f32_lanes>(0x7fff);
  const auto nan_floor = aie::broadcast<uint16_t, f32_lanes>(0x7f80);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < SILU_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    f32_acc xa;
    xa.from_vector(xv);
    const Limbs<f32_lanes> f{aie::max(aie::min(xv, top), bottom), zero, zero};
    const bf16_vec out = sigmoid_f32(xa, f).to_vector<bfloat16>();
    const auto pass =
        aie::gt(xv.cast_to<int16_t>(), top_bits) |
        aie::gt(aie::bit_and(xv.cast_to<uint16_t>(), magnitude), nan_floor);
    aie::store_v(y + i, aie::select(out, xv, pass));
  }
  aie::set_rounding(saved);
}

extern "C" {

void silu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  event0();
  silu_f32(input, output, input_size);
  event1();
}

} // extern "C"
