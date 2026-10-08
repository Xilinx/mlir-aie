//===- limbs_f32.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"
#include "../aie_kernel_utils.h"
#include "../common/f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LIMBS_ELEMS
#define LIMBS_ELEMS n
#endif

// AIE2 at 32 lanes trips Peano assertions (ISel, peephole) here.
constexpr int limbs_lanes = AIE_BF16_LANES;
using limbs_vec = aie::vector<bfloat16, limbs_lanes>;

// x as six planes of its bf16 limbs, (hi, mid, hi, lo, mid, hi). A bf16 matmul
// of them against a second operand's limbs stacked (hi, hi, mid, hi, mid, lo)
// sums the six largest of an f32 product's nine limb products.
extern "C" void limbs_f32(float *restrict x, bfloat16 *restrict y, int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const limbs_vec one = bf16_splat<limbs_lanes>(0x3f80);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < LIMBS_ELEMS; i += limbs_lanes) {
    limbs_vec hi, mid, lo;
    split3(aie::accum<accfloat, limbs_lanes>(aie::load_v<limbs_lanes>(x + i)),
           hi, mid, lo, one);
    aie::store_v(y + i, hi);
    aie::store_v(y + LIMBS_ELEMS + i, mid);
    aie::store_v(y + 2 * LIMBS_ELEMS + i, hi);
    aie::store_v(y + 3 * LIMBS_ELEMS + i, lo);
    aie::store_v(y + 4 * LIMBS_ELEMS + i, mid);
    aie::store_v(y + 5 * LIMBS_ELEMS + i, hi);
  }
  aie::set_rounding(saved);
  event1();
}
