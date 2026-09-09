//===- gelu.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// GELU (tanh approximation):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// 32-wide with MAC fusion and s*beta precompute to shorten the dependency
// chain:
//   inner1 = s*x + s_beta*x*x^2      (one MAC instead of mul+add+mul)
//   result = mac(0.5x, tanh, 0.5x)   (one MAC instead of add+mul+mul)
static inline aie::vector<bfloat16, 32>
gelu_tanh_approx(aie::vector<bfloat16, 32> x) {
  const bfloat16 k0_5 = 0.5f;
  const bfloat16 sqrt_2_over_pi = 0.79788456f;        // sqrt(2/pi)
  const bfloat16 s_beta = sqrt_2_over_pi * 0.044715f; // precomputed s*beta

  auto v05 = aie::broadcast<bfloat16, 32>(k0_5);
  auto vs2opi = aie::broadcast<bfloat16, 32>(sqrt_2_over_pi);
  auto vsBeta = aie::broadcast<bfloat16, 32>(s_beta);

  aie::vector<bfloat16, 32> x2 = aie::mul(x, x).to_vector<bfloat16>();
  aie::vector<bfloat16, 32> sbeta_x = aie::mul(x, vsBeta).to_vector<bfloat16>();
  auto sx = aie::mul(x, vs2opi);
  auto half_x = aie::mul(x, v05);

  auto inner1 = aie::mac(sx, sbeta_x, x2);
  auto tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());

  return aie::mac(half_x, tanh_out, half_x.to_vector<bfloat16>())
      .to_vector<bfloat16>();
}

// Out-of-place GELU: output_vector = gelu(input_vector).  input and output
// must not alias.
void gelu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                           bfloat16 *restrict output_vector,
                           const int32_t vector_size) {
  event0();
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  // AIE_PREPARE_FOR_POSTPIPELINING is required: the pre-RA pipeliner finds no
  // schedule for this body; the post-RA pipeliner achieves II=18, NS=2.
  auto body = [&]() __attribute__((always_inline)) {
    *it_out++ = gelu_tanh_approx(*it_in++);
  };
  VERSIONED_LOOP(2, (vector_size + 31) / 32, body,
                 AIE_PREPARE_FOR_POSTPIPELINING);
  event1();
}

// In-place GELU: v = gelu(v).  Single pointer, so aliasing-correct (each
// 32-lane slot is read then written).
static inline void gelu_tanh_approx_inplace_bf16(bfloat16 *restrict v,
                                                 const int32_t vector_size) {
  event0();
  auto it = aie::begin_restrict_vector<32>(v);
  auto body = [&]() __attribute__((always_inline)) {
    aie::vector<bfloat16, 32> x = *it;
    *it++ = gelu_tanh_approx(x);
  };
  VERSIONED_LOOP(2, (vector_size + 31) / 32, body,
                 AIE_PREPARE_FOR_POSTPIPELINING);
  event1();
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  int32_t input_size = 1024; // Assuming input size is a multiple of 32
  gelu_tanh_approx_bf16(input, output, input_size);
}

// In-place GELU over n bf16 elements (n a multiple of 32).  Intended as a fused
// epilogue over a compute tile (e.g. a GEMV output tile), applied once per tile
// in the producing core.
void gelu_tile_bf16(uint32_t n, bfloat16 *restrict c) {
  gelu_tanh_approx_inplace_bf16(c, (int32_t)n);
}

} // extern "C"
