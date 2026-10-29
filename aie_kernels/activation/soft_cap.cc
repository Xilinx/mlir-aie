//===- soft_cap.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SOFT_CAP
#error "SOFT_CAP (the cap, a float literal) must be defined"
#endif
#ifndef SOFT_CAP_ELEMS
#define SOFT_CAP_ELEMS n
#endif

// cap * tanh(x / cap) of bf16: x / cap and the product with cap in f32, the
// tanh tanh_f32's, rounded once to bf16. Past |x| = 64 cap, where tanh is 1 in
// f32, x is held there, so an infinite x gives +-cap. Below |x| = cap 2^-12,
// where the result rounds to x itself, and for NaN, x passes through.
extern "C" void soft_cap_bf16(bfloat16 *restrict x, bfloat16 *restrict y,
                              int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr float cap = SOFT_CAP;
  const Limbs<f32_lanes> inv_cap =
      bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(1.0f / cap)));
  const Limbs<f32_lanes> cap_limbs =
      bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(cap)));
  const bf16_vec top = aie::broadcast<bfloat16, f32_lanes>(64.0f * cap);
  const bf16_vec bottom = aie::broadcast<bfloat16, f32_lanes>(-64.0f * cap);
  const bf16_vec zero = aie::zeros<bfloat16, f32_lanes>();
  const auto magnitude = aie::broadcast<uint16_t, f32_lanes>(0x7fff);
  const auto nan_floor = aie::broadcast<uint16_t, f32_lanes>(0x7f80);
  const auto tiny =
      aie::broadcast<bfloat16, f32_lanes>(cap * 0x1p-12f).cast_to<uint16_t>();
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < SOFT_CAP_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    const bf16_vec held = aie::max(aie::min(xv, top), bottom);
    const Limbs<f32_lanes> xl{held, zero, zero};
    const f32_acc t = mac_limbs(aie::zeros<accfloat, f32_lanes>(), xl, inv_cap);
    const bf16_vec out = mac_limbs(aie::zeros<accfloat, f32_lanes>(),
                                   bf16_limbs(tanh_f32(t)), cap_limbs)
                             .to_vector<bfloat16>();
    const auto abs = aie::bit_and(xv.cast_to<uint16_t>(), magnitude);
    aie::store_v(
        y + i,
        aie::select(out, xv, aie::gt(abs, nan_floor) | aie::lt(abs, tiny)));
  }
  aie::set_rounding(saved);
  event1();
}
