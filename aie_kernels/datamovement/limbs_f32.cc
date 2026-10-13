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

#if AIE_TUNED_AIE2
// Storing a 32-lane plane whole trips Peano assertions (ISel, peephole)
// (llvm-aie#1372).
constexpr int limbs_lanes = 32;
constexpr int store_lanes = 16;
#else
constexpr int limbs_lanes = AIE_BF16_LANES;
constexpr int store_lanes = AIE_BF16_LANES;
#endif
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
    auto store_plane = [&](int p, const limbs_vec &v) {
      for (int h = 0; h < limbs_lanes / store_lanes; h++)
        aie::store_v(y + p * LIMBS_ELEMS + i + h * store_lanes,
                     v.extract<store_lanes>(h));
    };
    store_plane(0, hi);
    store_plane(1, mid);
    store_plane(2, hi);
    store_plane(3, lo);
    store_plane(4, mid);
    store_plane(5, hi);
  }
  aie::set_rounding(saved);
  event1();
}
