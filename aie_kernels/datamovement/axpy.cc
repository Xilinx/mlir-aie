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

// x * a + y with an f32 a: a is split, toward zero, into bf16 limbs that sum
// to it exactly and share its sign, and the nonzero ones are macced into y,
// smallest first. Each product is exact, so only the f32 sums round. A zero
// limb is left out, as its product with an infinite x would be NaN.
template <int N, int L>
static inline void saxpy_limbs(const bfloat16 *restrict x,
                               const bfloat16 *restrict y, bfloat16 *restrict z,
                               const uint16_t (&limb)[3], int steps) {
  ::aie::vector<bfloat16, L> aL[N];
  for (int i = 0; i < N; ++i)
    aL[i] = ::aie::broadcast<bfloat16, L>(
        __builtin_bit_cast(bfloat16, limb[N - 1 - i]));
  auto px = ::aie::begin_restrict_vector<L>(x);
  auto py = ::aie::begin_restrict_vector<L>(y);
  auto pz = ::aie::begin_restrict_vector<L>(z);
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  AIE_LOOP_NO_UNROLL
  for (int k = 0; k < steps; ++k) {
    ::aie::accum<accfloat, L> acc;
    acc.from_vector(*py++);
    const ::aie::vector<bfloat16, L> xv = *px++;
    for (int i = 0; i < N; ++i)
      acc = ::aie::mac(acc, xv, aL[i]);
    *pz++ = acc.template to_vector<bfloat16>();
  }
}

extern "C" {
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, const float a,
           bfloat16 *restrict z, const int32_t vector_size) {
  event0();
  // a's top 8 significant bits, the next 8, then the rest: truncating each
  // keeps every remainder exact and of a's sign, and a limb is zero only once
  // the remainder is.
  uint16_t limb[3];
  float rest = a;
  for (int i = 0; i < 3; ++i) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, rest) & 0xffff0000u;
    limb[i] = (uint16_t)(bits >> 16);
    rest = rest - __builtin_bit_cast(float, bits);
  }
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  constexpr int L = AIE_BF16_LANES;
#else
  constexpr int L = 64;
#endif
  // IRON only accepts a tile that is a multiple of 64.
  const int steps = (uint32_t)vector_size / L;
  if (limb[2] & 0x7fff)
    saxpy_limbs<3, L>(x, y, z, limb, steps);
  else if (limb[1] & 0x7fff)
    saxpy_limbs<2, L>(x, y, z, limb, steps);
  else
    saxpy_limbs<1, L>(x, y, z, limb, steps);
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
