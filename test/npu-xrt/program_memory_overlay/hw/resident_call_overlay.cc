//===- resident_call_overlay.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An overlay that calls back into the resident rather than being self-contained.
//
// The address of ovl_bias is resolved from the resident's symbol table when this
// overlay is linked, so nothing here or in the resident hardcodes it. That is
// the point of exporting the resident's defined symbols to each overlay link:
// code every overlay needs can live once, in the part of program memory that is
// always present, instead of being duplicated into every slot payload.
//
// Calling resident *code* fails differently from reading resident *data*. A data
// symbol bound to a wrong address corrupts memory; a code symbol bound to a
// wrong address sends the core into arbitrary bytes.

#include <cstdint>

extern "C" int32_t ovl_bias(void);

extern "C" void overlay_entry(int32_t *in, int32_t *out) {
  const int32_t bias = ovl_bias(); // resolved from the resident, = 100
  for (int i = 0; i < 256; i++)
    out[i] = in[i] + 77 + bias;
}
