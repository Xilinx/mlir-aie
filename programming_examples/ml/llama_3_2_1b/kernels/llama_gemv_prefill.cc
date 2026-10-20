//===- llama_gemv_prefill.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array's M-row (batched prefill) cores, in their own object so a
// core links only the code it runs.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

#ifndef LLAMA_GEMV_ARRAY_M
#define LLAMA_GEMV_ARRAY_M 1
#endif

// M-row (batched prefill) units: an activation object holds kM activations,
// each kMStride bytes [int8 x 2048 | act_scale f32 | pad to 64]; down's 8192
// columns arrive as four such objects, one per 2048-column chunk. A unit's
// output object holds kOutRows rows for each activation, activation-major,
// kOutRows * 4 bytes per activation (int8 gate rows use the first kOutRows).
// Every row is computed as the M=1 kernels compute it.
static constexpr int kM = LLAMA_GEMV_ARRAY_M;
static constexpr int kMStride = 2048 + 64;
static constexpr int kOutRows = 64;

extern "C" {

// One 4-row K=2048 slot against kM activations; `sic` is the slot's index
// within the output object.
void llama_gemv_array_m_f32(int8_t *restrict act, int8_t *restrict w,
                            int8_t *restrict out, int32_t sic) {
  event0();
  for (int m = 0; m < kM; m++)
    gemv_f32<2048, 4>(act + m * kMStride, w,
                      reinterpret_cast<float *>(out) + m * kOutRows + sic * 4);
  event1();
}

void llama_gemv_array_m_i8(int8_t *restrict act, int8_t *restrict w,
                           int8_t *restrict out, int32_t sic,
                           float inv_out_scale) {
  event0();
  for (int m = 0; m < kM; m++)
    gemv_i8<2048, 4>(act + m * kMStride, w, out + m * kOutRows * 4 + sic * 4,
                     inv_out_scale);
  event1();
}

// Down's first three column chunks: the 4-row slot's integer sums against
// each activation's chunk go into (chunk 0) or onto (chunks 1, 2) partial,
// int32[kM x rows], at rows slot*4..slot*4+3 (`rows`: the down rows this
// pass covers).
void llama_gemv_array_m_kchunk(int8_t *restrict act, int8_t *restrict w,
                               int32_t *restrict partial, int32_t slot,
                               int32_t chunk, int32_t rows) {
  event0();
  for (int m = 0; m < kM; m++)
    for (int n = 0; n < 4; n++) {
      int32_t s = gemv_row_sum<2048, 4>(act + m * kMStride, w + n * 2048, 0);
      int32_t *p = partial + m * rows + slot * 4 + n;
      *p = chunk == 0 ? s : *p + s;
    }
  event1();
}

// Down's last chunk: the sums complete with the slot's bias, then the f32
// epilogue with each activation's scale.
void llama_gemv_array_m_kfinal(int8_t *restrict act, int8_t *restrict w,
                               int32_t *restrict partial, int8_t *restrict out,
                               int32_t slot, int32_t sic, int32_t rows) {
  event0();
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + 4 * 2048);
  const float *w_scales = reinterpret_cast<const float *>(w + 4 * 2048 + 16);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int m = 0; m < kM; m++) {
    const int8_t *a = act + m * kMStride;
    int32_t sums[4];
    for (int n = 0; n < 4; n++)
      sums[n] = partial[m * rows + slot * 4 + n] +
                gemv_row_sum<2048, 4>(a, w + n * 2048, bias[n]);
    epilogue_f32<4>(sums, *reinterpret_cast<const float *>(a + 2048), w_scales,
                    reinterpret_cast<float *>(out) + m * kOutRows + sic * 4);
  }
  event1();
}

} // extern "C"
