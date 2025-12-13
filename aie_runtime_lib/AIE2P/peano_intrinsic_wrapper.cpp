//===- peano_intrinsic_wrapper.cpp ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// Include lut_based_ops data declarations
alignas(aie::vector_decl_align) extern int16 exp_ilut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_ilut_cd[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_cd[512];
alignas(aie::vector_decl_align) extern unsigned char m_inv_lut[128];
extern float tanh_lut_ab[];
extern float tanh_lut_cd[];

// Include the C++ implementations
#include "lut_based_ops.h"
#include "vec_math.h"

// Provide extern "C" wrapper functions that call the C++ implementations
// These have C linkage (no name mangling) for LLVM backend linking
extern "C" {

// LUT-based operations
v16accfloat getExpBf16_wrapper(v16bfloat16 in) { return getExpBf16(in); }

bfloat16 getInvBf16_wrapper(float in) { return getInvBf16(in); }

v16bfloat16 getTanhBf16_wrapper(v16bfloat16 in) { return getTanhBf16(in); }

// Vector math operations - v16 versions
v16bfloat16 getRsqrtBf16_wrapper(v16bfloat16 in) { return getRsqrtBf16(in); }

v16bfloat16 getSqrtBf16_wrapper(v16bfloat16 in) { return getSqrtBf16(in); }

v16bfloat16 getErfBf16_wrapper(v16bfloat16 in) { return getErfBf16(in); }

v16bfloat16 getSigmoidBf16_wrapper(v16bfloat16 in) {
  return getSigmoidBf16(in);
}

v16bfloat16 getCeilBf16_wrapper(v16bfloat16 in) { return getCeilBf16(in); }

v16bfloat16 getFloorBf16_wrapper(v16bfloat16 in) { return getFloorBf16(in); }

// Vector math operations - v32 versions
v32bfloat16 getRsqrtBf16_v32_wrapper(v32bfloat16 in) {
  return getRsqrtBf16(in);
}

v32bfloat16 getSqrtBf16_v32_wrapper(v32bfloat16 in) { return getSqrtBf16(in); }

v32bfloat16 getErfBf16_v32_wrapper(v32bfloat16 in) { return getErfBf16(in); }

v32bfloat16 getSigmoidBf16_v32_wrapper(v32bfloat16 in) {
  return getSigmoidBf16(in);
}

v32bfloat16 getCeilBf16_v32_wrapper(v32bfloat16 in) { return getCeilBf16(in); }

v32bfloat16 getFloorBf16_v32_wrapper(v32bfloat16 in) {
  return getFloorBf16(in);
}

// getAbs overloads
v32bfloat16 getAbs_v32bf16_wrapper(v32bfloat16 in) { return getAbs(in); }

v16bfloat16 getAbs_v16bf16_wrapper(v16bfloat16 in) { return getAbs(in); }

v16float getAbs_v16f_wrapper(v16float in) { return getAbs(in); }

v16int32 getAbs_v16i32_wrapper(v16int32 in) { return getAbs(in); }

v32int16 getAbs_v32i16_wrapper(v32int16 in) { return getAbs(in); }

v64int8 getAbs_v64i8_wrapper(v64int8 in) { return getAbs(in); }

} // extern "C"
