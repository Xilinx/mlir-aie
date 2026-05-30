//===- llama_gemm_int8_srs.cc ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama gemm_int8_srs v0: GEMV path for decode (M=1), single per-tensor
// right_shift, no per-channel weight scales yet. Mirrors the yolo
// m10_linear_gemm pattern -- bias-init the acc, vec mac across K,
// reduce_add per output channel, banker_srs scalar tail, clamp.
//
//   acc[n]  = sum_k act[k] * weights[n*K + k]  (int8 * int8 -> int32)
//   sum[n]  = acc[n] + bias[n]
//   out[n]  = banker_srs(sum[n], right_shift)  saturated to int8
//
// w_packed layout: [weights : int8[N*K]] || [bias : int32[N]] (packed via
// reinterpret cast at the K*N byte boundary). This is the cautious-eureka
// StaticWeightStream pattern: per-call constants delivered as one packed
// payload over a single ObjectFifo (fits the 2-in CT DMA budget).
//
// Per-channel weight scales + per-token act scales are a v1 follow-up.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef LLAMA_GEMM_K
#define LLAMA_GEMM_K 64
#endif
#ifndef LLAMA_GEMM_N
#define LLAMA_GEMM_N 64
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

// Round-half-to-even with positive-bias offset; matches the scalar
// banker_srs used in yolo m10_linear_gemm.
static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void llama_gemm_int8_srs(int8_t *act, int8_t *w_packed, int8_t *out,
                         int32_t right_shift) {
  event0();

  constexpr int kK = LLAMA_GEMM_K;
  constexpr int kN = LLAMA_GEMM_N;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = w_packed;
  const int32_t *bias = reinterpret_cast<const int32_t *>(w_packed + kN * kK);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  AIE_LOOP_RANGE(kN, kN)
  for (int n = 0; n < kN; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    AIE_LOOP_RANGE(kGroups, kGroups)
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }

    int32_t sum = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    out[n] = (int8_t)s;
  }

  event1();
}

} // extern "C"
