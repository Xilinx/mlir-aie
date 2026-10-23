//===- tanh.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef TANH_ELEMS
#define TANH_ELEMS n
#endif

// tanh of bf16, tanh_f32's, rounded once to bf16. Below |x| = 2^-12, where the
// result rounds to x itself, and for NaN, x passes through.
extern "C" void tanh_bf16(bfloat16 *restrict x, bfloat16 *restrict y,
                          int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const auto magnitude = aie::broadcast<uint16_t, f32_lanes>(0x7fff);
  const auto nan_floor = aie::broadcast<uint16_t, f32_lanes>(0x7f80);
  const auto tiny = aie::broadcast<uint16_t, f32_lanes>(0x3980);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < TANH_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    f32_acc t;
    t.from_vector(xv);
    const bf16_vec out = tanh_f32(t).to_vector<bfloat16>();
    const auto abs = aie::bit_and(xv.cast_to<uint16_t>(), magnitude);
    aie::store_v(
        y + i,
        aie::select(out, xv, aie::gt(abs, nan_floor) | aie::lt(abs, tiny)));
  }
  aie::set_rounding(saved);
  event1();
}
