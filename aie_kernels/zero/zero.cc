//===- zero.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/zero.h"

extern "C" void zero(ZERO_TYPE *__restrict output) {
  event0();
#ifdef ZERO_SCALAR
  zero_scalar<ZERO_TYPE, TILE_SIZE, 1>(output);
#else
  zero_vectorized<ZERO_TYPE, TILE_SIZE, 1, false>(output);
#endif
  event1();
}

extern "C" void zero_window(ZERO_TYPE *__restrict output, int32_t offset,
                            int32_t count) {
  event0();
  zero_vectorized(output + offset, count);
  event1();
}
