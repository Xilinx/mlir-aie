//===- clamp.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef CLAMP_ELEMS
#define CLAMP_ELEMS n
#endif

// Clamp: f(x) = min(max(x, low), high). Each bound arrives as its bf16 bits in
// an int32, the type a runtime parameter word holds.
extern "C" void clamp_bf16(bfloat16 *restrict x, bfloat16 *restrict y,
                           int32_t n, int32_t low, int32_t high) {
  event0();
  auto it_in = aie::begin_restrict_vector<32>(x);
  auto it_out = aie::begin_restrict_vector<32>(y);
  const aie::vector<bfloat16, 32> lo =
      aie::broadcast<bfloat16, 32>(__builtin_bit_cast(bfloat16, (int16_t)low));
  const aie::vector<bfloat16, 32> hi =
      aie::broadcast<bfloat16, 32>(__builtin_bit_cast(bfloat16, (int16_t)high));
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < CLAMP_ELEMS; i += 32)
    *it_out++ = aie::min(aie::max(*it_in++, lo), hi);
  event1();
}
