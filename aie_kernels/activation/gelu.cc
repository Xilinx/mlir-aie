//===- gelu.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GELU_ELEMS
#define GELU_ELEMS n
#endif

// GELU's tanh form of bf16, 0.5 x (1 + tanh(u)) with
// u = sqrt(2/pi) (x + 0.044715 x^3), as x sigmoid(2u): sigmoid_f32 with x for
// its factor, rounded once to bf16. 2u = x (c + d x^2) in f32, x^2 exact. x is
// held at +-16 first, so -inf gives -0; above 16, where sigmoid(2u) is 1 in
// f32, and for NaN, x passes through. x and y may alias.
static inline void gelu_f32(const bfloat16 *x, bfloat16 *y, int32_t n) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec top = aie::broadcast<bfloat16, f32_lanes>(16.0f);
  const bf16_vec bottom = aie::broadcast<bfloat16, f32_lanes>(-16.0f);
  const f32_acc c(aie::broadcast<float, f32_lanes>(1.5957691216057308f));
  const Limbs<f32_lanes> d = bf16_limbs(
      f32_acc(aie::broadcast<float, f32_lanes>(0.07135481627159783f)));
  const bf16_vec zero = aie::zeros<bfloat16, f32_lanes>();
  const auto top_bits = aie::broadcast<int16_t, f32_lanes>(0x4180);
  const auto magnitude = aie::broadcast<uint16_t, f32_lanes>(0x7fff);
  const auto nan_floor = aie::broadcast<uint16_t, f32_lanes>(0x7f80);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < GELU_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    const bf16_vec held = aie::max(aie::min(xv, top), bottom);
    const Limbs<f32_lanes> inner =
        bf16_limbs(mac_limbs(c, bf16_limbs(aie::mul(held, held)), d));
    const f32_acc v = aie::mac(
        aie::mac(aie::mul(held, inner.hi), held, inner.mid), held, inner.lo);
    const bf16_vec out = sigmoid_f32(v, Limbs<f32_lanes>{held, zero, zero})
                             .to_vector<bfloat16>();
    const auto pass =
        aie::gt(xv.cast_to<int16_t>(), top_bits) |
        aie::gt(aie::bit_and(xv.cast_to<uint16_t>(), magnitude), nan_floor);
    aie::store_v(y + i, aie::select(out, xv, pass));
  }
  aie::set_rounding(saved);
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  event0();
  gelu_f32(input, output, 1024);
  event1();
}

void gelu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  event0();
  gelu_f32(input, output, input_size);
  event1();
}

// In place over n bf16 elements, n a multiple of 32: a fused epilogue over a
// compute tile, such as a GEMV's output, applied once per tile in the core
// that produced it.
void gelu_tile_bf16(uint32_t n, bfloat16 *restrict c) {
  event0();
  gelu_f32(c, c, (int32_t)n);
  event1();
}

} // extern "C"
