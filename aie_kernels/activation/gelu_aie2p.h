//===- gelu_aie2p.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifdef GELU_ELEMS
static_assert(GELU_ELEMS > 0 && GELU_ELEMS % 32 == 0);
#else
#define GELU_ELEMS vector_size
#endif

// Tile size of gelu_bf16, which takes no size argument.
constexpr int32_t gelu_tile_elems = 1024;

// GELU (tanh approximation):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// 32-wide with MAC fusion and s*beta precompute to shorten the dependency
// chain:
//   inner1 = s*x + s_beta*x*x^2      (one MAC instead of mul+add+mul)
//   result = mac(0.5x, tanh, 0.5x)   (one MAC instead of add+mul+mul)
//
// 1 + tanh is 0 from x = -8 down, so 0.5x is clamped there rather than making
// -inf * 0. tanh(-inf) is already -1, so the clamp stays off its input, where
// it costs 35 cycles a tile rather than 19.
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
  auto half_x = aie::mul(aie::max(x, bfloat16(-8.0f)), v05);

  auto inner1 = aie::mac(sx, sbeta_x, x2);
  auto tanh_out = tanh_bf16_vec<32>(inner1.to_vector<float>());

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
  // schedule for this body.
  // Unrolling fills the mac-to-vtanh stall; by eight it runs out of registers.
  auto body = [&]() __attribute__((always_inline)) {
    *it_out++ = gelu_tanh_approx(*it_in++);
  };
  AIE_PREPARE_FOR_POSTPIPELINING
  AIE_LOOP_UNROLL(4)
  for (int i = 0; i < GELU_ELEMS; i += 32)
    body();
  event1();
}

// In-place GELU: v = gelu(v).  Single pointer, so aliasing-correct (each
// 32-lane slot is read then written).
static inline void gelu_tanh_approx_inplace_bf16(bfloat16 *restrict v,
                                                 const int32_t vector_size) {
  event0();
  // Separate read and write cursors, so the loads and stores are separate
  // streams. Both derive from `v`: a second begin_restrict_vector would claim
  // the streams are disjoint, which in place they are not.
  auto it_in = aie::begin_vector<32>(v);
  auto it_out = aie::begin_vector<32>(v);
  auto body = [&]() __attribute__((always_inline)) {
    *it_out++ = gelu_tanh_approx(*it_in++);
  };
  VERSIONED_LOOP(2, (vector_size + 31) / 32, body,
                 AIE_PREPARE_FOR_POSTPIPELINING);
  event1();
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  gelu_tanh_approx_bf16(input, output, gelu_tile_elems);
}

void gelu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  gelu_tanh_approx_bf16(input, output, input_size);
}

// In-place GELU over n bf16 elements (n a multiple of 32).  Intended as a fused
// epilogue over a compute tile (e.g. a GEMV output tile), applied once per tile
// in the producing core.
void gelu_tile_bf16(uint32_t n, bfloat16 *restrict c) {
  gelu_tanh_approx_inplace_bf16(c, (int32_t)n);
}

} // extern "C"
