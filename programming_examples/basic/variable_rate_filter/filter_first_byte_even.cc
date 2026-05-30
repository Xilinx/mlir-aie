//===- filter_first_byte_even.cc -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//
// Window-copy kernel called on the forwarded iterations of the
// VariableRateFifo example. The skip decision is made at the IRON
// Python layer via a Python-level alternating pattern; this kernel
// is invoked only on iterations the producer chose to forward, and
// simply copies the input window to the output buffer.
//===----------------------------------------------------------------------===//

#include <stdint.h>

extern "C" {

void filterFirstByteEven(uint8_t *in_window, uint8_t *out_window,
                         int32_t line_size) {
  for (int32_t i = 0; i < line_size; ++i) {
    out_window[i] = in_window[i];
  }
}

}  // extern "C"
