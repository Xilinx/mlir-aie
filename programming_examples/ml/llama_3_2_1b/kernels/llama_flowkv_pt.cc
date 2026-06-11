//===- llama_flowkv_pt.cc -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 1.7 dataflow stubs for the FlowKV qk -> sv pair. Both kernels
// bitwise-invert their input (out[i] = ~in[i]). Composed, the pair
// returns the original input -- so the host can bit-exact verify that
// BOTH stubs ran in sequence (single failure or shim-bypass would
// produce ~input instead).
//
// Real kernels: flowkv_qk runs on CT0 (Q @ K^T chunk + reduce_max +
// linear_approx exp + reduce_add -> softmax stats); flowkv_sv runs on
// CT1 (online-softmax merge of correction + V acc -> normalized output).
// These stubs share a kernel object (same .o) under two different
// extern "C" symbols so the IRON design can declare them separately.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#ifndef LLAMA_FLOWKV_PT_BYTES
#define LLAMA_FLOWKV_PT_BYTES 512
#endif

extern "C" {

void llama_flowkv_qk_pt(int8_t *in, int8_t *out) {
  for (int i = 0; i < LLAMA_FLOWKV_PT_BYTES; i++) {
    out[i] = ~in[i];
  }
}

void llama_flowkv_sv_pt(int8_t *in, int8_t *out) {
  for (int i = 0; i < LLAMA_FLOWKV_PT_BYTES; i++) {
    out[i] = ~in[i];
  }
}

} // extern "C"
