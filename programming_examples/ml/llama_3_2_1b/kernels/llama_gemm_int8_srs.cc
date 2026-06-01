//===- llama_gemm_int8_srs.cc ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama gemm_int8_srs v0: GEMV path for decode (M=1), single per-tensor
// right_shift, no per-channel weight scales yet. Mirrors yolo
// m10_linear_gemm.
//
// One template implementation, multiple shape-specific extern "C"
// entries so the same .cc/.o serves every projection call site in a
// layer (q/o use D->D, gate/up use D->HD, down uses HD->D). IRON's
// Kernel(symbol, .o, types) declaration requires per-shape C symbols
// since the arg types differ per call site.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Llama 3.2 1B integration test sizes (small for first bring-up).
static constexpr int kD  = 64;
static constexpr int kHD = 256;
#ifndef LLAMA_GEMM_VOCAB
#define LLAMA_GEMM_VOCAB 256        // first-cut lm_head vocab (Phase 6b)
#endif
static constexpr int kV  = LLAMA_GEMM_VOCAB;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

template <int kK, int kN>
static inline void gemm_impl(int8_t *act, int8_t *w_packed, int8_t *out,
                             int32_t right_shift) {
  event0();
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = w_packed;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(w_packed + kN * kK);

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

extern "C" {

// Shape-specific entry points for the per-layer integration design.
void llama_gemm_int8_srs_D_to_D(int8_t *act, int8_t *w, int8_t *out, int32_t rs) {
  gemm_impl<kD, kD>(act, w, out, rs);
}
void llama_gemm_int8_srs_D_to_HD(int8_t *act, int8_t *w, int8_t *out, int32_t rs) {
  gemm_impl<kD, kHD>(act, w, out, rs);
}
void llama_gemm_int8_srs_HD_to_D(int8_t *act, int8_t *w, int8_t *out, int32_t rs) {
  gemm_impl<kHD, kD>(act, w, out, rs);
}
// lm_head: D-dim activation -> V-dim int8 logits.
void llama_gemm_int8_srs_D_to_V(int8_t *act, int8_t *w, int8_t *out, int32_t rs) {
  gemm_impl<kD, kV>(act, w, out, rs);
}

// Compat alias used by the standalone gemm bring-up test
// (test_gemm_int8_srs_real.py at K=N=64).
void llama_gemm_int8_srs(int8_t *act, int8_t *w, int8_t *out, int32_t rs) {
  gemm_impl<kD, kD>(act, w, out, rs);
}

} // extern "C"
