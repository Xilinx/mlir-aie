//===- filter_first_byte_even.cc -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// G-T3.2-007 worked example kernel.
//
// Predicate: forward the input window iff the first byte is even.
// On forward, copy the input window to the output buffer; on
// skip, leave the output buffer untouched (the variable-rate
// fifo's discard semantic means the consumer never sees this
// slot anyway).
//
// Returns 1 on forward, 0 on skip. The IRON Python topology
// receives the return as a runtime branch hint; the actual
// "skip" semantic at the lock layer is realized by the Python
// kernel calling `out_handle.discard(1)` in the skip branch
// (instead of `out_handle.acquire(1) + ... + release(1)`).
//===----------------------------------------------------------------------===//

#include <stdint.h>

extern "C" {

int32_t filterFirstByteEven(uint8_t *in_window, uint8_t *out_window,
                            int32_t line_size) {
  // Predicate: first byte is even.
  if ((in_window[0] & 0x1) != 0) {
    // Skip. Returning 0 signals "discard". The IRON kernel
    // function uses the return code to branch into discard(1).
    return 0;
  }

  // Forward. Copy the input window to the output buffer.
  for (int32_t i = 0; i < line_size; ++i) {
    out_window[i] = in_window[i];
  }
  return 1;
}

}  // extern "C"
