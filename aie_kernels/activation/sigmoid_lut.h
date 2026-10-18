//===- sigmoid_lut.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../aie_kernel_utils.h"
#include "../common/activations.h"
#include <aie_api/aie.hpp>

#if (AIE_TUNED_AIE2 || AIE_TUNED_AIE2P) && !ACTIVATIONS_NATIVE_TANH
#include "aie_bank_placement.h"
// The LUT sigmoid reads its own table rather than tanh's: tanh_lut_ab/cd
// with each segment rewritten for 0.5 + 0.5 * tanh(x/2), the slope over 4 and
// the offset 0.5 + 0.5 * offset, indexed by x in segments of 0.5 over [-8, 8).
// That drops the x/2 and 0.5 * (1 + t) passes around the table reads and one
// of the two bf16 roundings. Every slope is still a bf16, which the lookup
// reads as the high half of each float.
// Each bank run of AIE_LUT_16B_RUN uint16 holds AIE_LUT_16B_RUN / 4 {slope,
// offset} pairs, and the lookup wants every run stored twice, as tanh_lut_ab/cd
// are; the two banks hold the same table. Each row below is four pairs, left to
// right.
#if AIE_LUT_16B_RUN == 8
#define SIGMOID_LUT_ROW(s0, o0, s1, o1, s2, o2, s3, o3)                        \
  s0, o0, s1, o1, s0, o0, s1, o1, s2, o2, s3, o3, s2, o2, s3, o3
#else
#define SIGMOID_LUT_ROW(...) __VA_ARGS__, __VA_ARGS__
#endif
// clang-format off
#define SIGMOID_LUT_TABLE {                           \
  SIGMOID_LUT_ROW(0.0f, 0.0f,                         \
                  0.00070953369140625f, 0.005859375f, \
                  0.00127410888671875f, 0.009765625f, \
                  0.0018768310546875f, 0.013671875f), \
  SIGMOID_LUT_ROW(0.003173828125f, 0.021484375f,      \
                  0.00531005859375f, 0.033203125f,    \
                  0.00885009765625f, 0.05078125f,     \
                  0.01409912109375f, 0.07421875f),    \
  SIGMOID_LUT_ROW(0.02294921875f, 0.109375f,          \
                  0.036376953125f, 0.15625f,          \
                  0.057373046875f, 0.21875f,          \
                  0.0869140625f, 0.2919921875f),      \
  SIGMOID_LUT_ROW(0.1259765625f, 0.3701171875f,       \
                  0.1728515625f, 0.440185546875f,     \
                  0.216796875f, 0.484619140625f,      \
                  0.25f, 0.5f),                       \
  SIGMOID_LUT_ROW(0.25f, 0.5f,                        \
                  0.216796875f, 0.515380859375f,      \
                  0.1728515625f, 0.559814453125f,     \
                  0.1259765625f, 0.6298828125f),      \
  SIGMOID_LUT_ROW(0.0869140625f, 0.7080078125f,       \
                  0.057373046875f, 0.78125f,          \
                  0.036376953125f, 0.84375f,          \
                  0.02294921875f, 0.890625f),         \
  SIGMOID_LUT_ROW(0.01409912109375f, 0.92578125f,     \
                  0.00885009765625f, 0.94921875f,     \
                  0.00531005859375f, 0.966796875f,    \
                  0.003173828125f, 0.978515625f),     \
  SIGMOID_LUT_ROW(0.0018768310546875f, 0.986328125f,  \
                  0.00127410888671875f, 0.990234375f, \
                  0.00070953369140625f, 0.994140625f, \
                  0.0f, 1.0f)                         \
}
// clang-format on
// inline: sigmoid.cc, silu.cc and swiglu.cc each define the tables, so one
// core linking two of them keeps a single copy rather than a duplicate symbol.
AIE_BANK_A alignas(aie::vector_decl_align) inline float sigmoid_lut_ab[128] =
    SIGMOID_LUT_TABLE;
AIE_BANK_B alignas(aie::vector_decl_align) inline float sigmoid_lut_cd[128] =
    SIGMOID_LUT_TABLE;
#undef SIGMOID_LUT_TABLE
#undef SIGMOID_LUT_ROW

__attribute__((always_inline)) inline aie::vector<bfloat16, 16>
sigmoid_lut_bf16(aie::vector<bfloat16, 16> x) {
  return lut_segments_acc<5>(sigmoid_lut_ab, sigmoid_lut_cd, x)
      .to_vector<bfloat16>();
}
#endif
