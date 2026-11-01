// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Scalar transpose with mb, nb as ordinary int32 arguments instead of the
// -DDIM_m/-DDIM_n macros transpose_4x4/transpose_8x8 need, so one compiled
// object serves any (mb, nb), with no divisibility or minimum-size
// constraint, at one element per cycle instead of a VSHUFFLE per tile.

#include <cstdint>

#if !defined(DTYPE_i8) && !defined(DTYPE_i16) && !defined(DTYPE_i32)
#error Please specify data type at kernel compile time using e.g., -DDTYPE_i8 or -DDTYPE_i16 or -DDTYPE_i32.
#endif

#if defined(DTYPE_i8)
#define DTYPE uint8_t
#endif
#if defined(DTYPE_i16)
#define DTYPE uint16_t
#endif
#if defined(DTYPE_i32)
#define DTYPE uint32_t
#endif

extern "C" {

void transpose_dyn(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr,
                   int32_t mb, int32_t nb) {
  for (int32_t i = 0; i < mb; i++) {
    const DTYPE *in_row = in_ptr + i * nb;
    for (int32_t j = 0; j < nb; j++) {
      out_ptr[j * mb + i] = in_row[j];
    }
  }
}
}
