//===- gelu_aie2.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifdef GELU_ELEMS
static_assert(GELU_ELEMS > 0 && GELU_ELEMS % 16 == 0);
#else
#define GELU_ELEMS vector_size
#endif

#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

// Tile size of gelu_bf16, which takes no size argument.
constexpr int32_t gelu_tile_elems = 1024;

// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), as u = x (c + d x^2) and
// 0.5 x + 0.5 x tanh(u), each accumulated in fp32.
static inline __attribute__((always_inline)) aie::vector<bfloat16, 16>
gelu_v16(aie::vector<bfloat16, 16> x) {
  const float sqrt_2_over_pi = 0.79788456f;
  const float kBeta = 0.044715f;
  aie::accum<accfloat, 16> c(aie::broadcast<float, 16>(sqrt_2_over_pi));
  auto d = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi * kBeta);
  auto v05 = aie::broadcast<bfloat16, 16>(0.5f);

  aie::vector<bfloat16, 16> xl = aie::max(x, bfloat16(-8.0f));
  aie::vector<bfloat16, 16> x2 = aie::mul(xl, xl).to_vector<bfloat16>();
  aie::vector<bfloat16, 16> p = aie::mac(c, x2, d).to_vector<bfloat16>();
  aie::vector<bfloat16, 16> u = aie::mul(xl, p).to_vector<bfloat16>();
  aie::vector<bfloat16, 16> t = getTanhBf16(u);
  aie::accum<accfloat, 16> hx_acc = aie::mul(xl, v05);
  aie::vector<bfloat16, 16> hx = hx_acc.to_vector<bfloat16>();
  return aie::mac(hx_acc, t, hx).to_vector<bfloat16>();
}

void gelu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                           bfloat16 *restrict output_vector,
                           const int32_t vector_size) {
  event0();

  lut_map_bf16(input_vector, output_vector, GELU_ELEMS, gelu_v16);

  event1();

  return;
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  gelu_tanh_approx_bf16(input, output, gelu_tile_elems);
}

void gelu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  gelu_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
