//===- decode_gate_layer_embedding_core.cc ----------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Lock-free test entry point around decode_gate_layer_embedding.cc's
// activation, which works in place: it runs on a copy of x in y.

#include "decode_gate_layer_embedding.cc"

extern "C" void pli_gelu_core(bf16 *x, bf16 *y) {
  event0();
  copy_vectorized<bf16, PLI_D>(y, x);
  _activate(y);
  event1();
}
