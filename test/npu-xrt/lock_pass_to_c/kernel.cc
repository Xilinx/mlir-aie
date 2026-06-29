//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This kernel demonstrates receiving lock IDs as function arguments and using
// them to perform acquire/release operations in C code. The lock IDs are
// localized constant integers passed from MLIR.

#include <stdint.h>

extern "C" {

void scale_with_locks(int32_t *in, int32_t *out, int64_t in_cons_lock,
                      int64_t in_prod_lock, int64_t out_prod_lock,
                      int64_t out_cons_lock) {
  // Acquire input consumer lock — wait for input data ready
  acquire_equal(in_cons_lock, -1);

  // Acquire output producer lock — wait for output buffer free
  acquire_equal(out_prod_lock, -1);

  // Scale each element by 3
  for (int i = 0; i < 1024; i++) {
    out[i] = in[i] * 3;
  }

  // Release input producer lock — signal input buffer free
  release(in_prod_lock, 1);

  // Release output consumer lock — signal output data ready
  release(out_cons_lock, 1);
}

} // extern "C"
