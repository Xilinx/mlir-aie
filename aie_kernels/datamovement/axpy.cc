//===- axpy.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

// 64 / AIE_BF16_LANES, as a literal for the loop pragma.
#if AIE_TUNED_AIE2P
#define AXPY_BLOCK_VECTORS 2
#else
#define AXPY_BLOCK_VECTORS 4
#endif

extern "C" {
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, const float a,
           bfloat16 *restrict z, const int32_t vector_size) {
  event0();
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  // y loads straight into the accumulator (vlda.conv, a port) and x on the b
  // port, so each vector is one mac and one converting store. The deep
  // pipelined schedule needs the promised 16 trips; a shorter tile takes a loop
  // that promises one 64-element block.
  constexpr int L = AIE_BF16_LANES;
  ::aie::vector<bfloat16, L> aL = ::aie::broadcast<bfloat16, L>(bfloat16(a));
  auto px = ::aie::begin_restrict_vector<L>(x);
  auto py = ::aie::begin_restrict_vector<L>(y);
  auto pz = ::aie::begin_restrict_vector<L>(z);
  const int steps = (uint32_t)vector_size / 64 * (64 / L);
  if (steps >= 16) {
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < steps; ++k) {
      ::aie::accum<accfloat, L> acc;
      acc.from_vector(*py++);
      *pz++ = ::aie::mac(acc, *px++, aL).template to_vector<bfloat16>();
    }
  } else if (steps > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(AXPY_BLOCK_VECTORS)
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < steps; ++k) {
      ::aie::accum<accfloat, L> acc;
      acc.from_vector(*py++);
      *pz++ = ::aie::mac(acc, *px++, aL).template to_vector<bfloat16>();
    }
  }
#else
  ::aie::vector<bfloat16, 64> a_v = ::aie::broadcast<bfloat16, 64>(bfloat16(a));
  // IRON only accepts a tile that is a multiple of 64; unsigned, the divide is
  // a shift.
  const int steps = (uint32_t)vector_size / 64;
  if (steps > 0) {
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int k = 0; k < steps; ++k) {
      ::aie::vector<bfloat16, 64> x_v = ::aie::load_v<64>(x);
      x += 64;
      ::aie::vector<bfloat16, 64> y_v = ::aie::load_v<64>(y);
      y += 64;
      ::aie::accum<accfloat, 64> ax_v = ::aie::mul(x_v, a_v);
      ::aie::accum<accfloat, 64> z_v = ::aie::add(ax_v, y_v);
      ::aie::vector<bfloat16, 64> z_v_converted = z_v.to_vector<bfloat16>();
      ::aie::store_v(z, z_v_converted);
      z += 64;
    }
  }
#endif
  event1();
}

void saxpy_scalar(bfloat16 *x, bfloat16 *y, const bfloat16 a, bfloat16 *z,
                  const int32_t vector_size) {
  event0();
  float a_f = a;
  for (int i = 0; i < vector_size; ++i) {
    z[i] = a_f * x[i] + y[i];
  }
  event1();
}
}
