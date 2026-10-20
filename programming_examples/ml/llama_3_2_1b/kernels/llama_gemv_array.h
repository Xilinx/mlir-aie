//===- llama_gemv_array.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// What the GEMV array's kernels (llama_gemv_array.cc, llama_gemv_mma.cc)
// share: the int8 bounds and the vector fp32 forms.
//===----------------------------------------------------------------------===//

#ifndef LLAMA_GEMV_ARRAY_H
#define LLAMA_GEMV_ARRAY_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

#endif // LLAMA_GEMV_ARRAY_H
