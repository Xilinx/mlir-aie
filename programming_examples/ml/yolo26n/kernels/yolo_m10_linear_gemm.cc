//===- yolo_m10_linear_gemm.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Scalar i8 Gemm 1280 → 2 for the yolo26n-cls binary classifier head.
// Weights are stored in raw `shape_2x1280` layout (row-major, no OIYX
// packing — manifest weights_layout="shape_2x1280"). Each output is
//   out[o] = SRS_i8(sum_d wts[o, d] * in[d] + bias[o], right_shift=10)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m10_linear_gemm_i8_i8(int8_t *in_vec, // (in_dim,)
                                int8_t *wts, // (out_dim, in_dim) row-major flat
                                int32_t *bias,               // (out_dim,)
                                int8_t *out_vec,             // (out_dim,)
                                const int32_t in_dim,        // 1280
                                const int32_t out_dim,       // 2
                                const int32_t right_shift) { // 10
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // Hardcoded for m10 head (in_dim=1280, out_dim=2, rs=10). Per-output:
  // 1280/64 = 20 vec-mac iters over 64-lane i8 strips, one reduce_add,
  // one scalar SRS+clamp. Replaces the 1280-iter scalar dot product.
  (void)in_dim;
  (void)out_dim;
  constexpr int kInDim = 1280;
  constexpr int kOutDim = 2;
  constexpr int kVec = 64;
  constexpr int kGroups = kInDim / kVec; // 20

  AIE_LOOP_RANGE(kOutDim, kOutDim)
  for (int o = 0; o < kOutDim; o++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = wts + o * kInDim;
    AIE_LOOP_RANGE(kGroups, kGroups)
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(in_vec + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }

    int32_t sum = aie::reduce_add(acc.template to_vector<int32>()) + bias[o];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX)
      s = I8_MAX;
    if (s < I8_MIN)
      s = I8_MIN;
    out_vec[o] = (int8_t)s;
  }

  event1();
}

} // extern "C"
