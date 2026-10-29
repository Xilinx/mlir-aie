//===- sigmoid.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SIGMOID_ELEMS
#define SIGMOID_ELEMS n
#endif

// sigmoid of bf16, sigmoid_f32's, rounded once to bf16.
extern "C" void sigmoid_bf16(bfloat16 *restrict x, bfloat16 *restrict y,
                             int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < SIGMOID_ELEMS; i += f32_lanes) {
    f32_acc xa;
    xa.from_vector(aie::load_v<f32_lanes>(x + i));
    aie::store_v(y + i, sigmoid_f32(xa).to_vector<bfloat16>());
  }
  aie::set_rounding(saved);
  event1();
}
