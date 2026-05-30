//===- llama_glue_pt.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 1.8 dataflow stub for multi-input glue tiles (rmsnorm+residual,
// silu+mul style). Two inputs converge at one compute tile; output is
// the bitwise xor of the two inputs. Host bit-exact verifies the
// 2-input fanin pattern works.
//
// Real kernels with this fanin shape:
//   - llama_rmsnorm_residual: normed(in1) + residual(in2) fused
//   - llama_silu_mul:         silu(gate=in1) * up(in2)
//   - llama_flowkv_sv:        softmax_stats(in1) + V_chunk(in2) merge
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#ifndef LLAMA_GLUE_PT_BYTES
#define LLAMA_GLUE_PT_BYTES 512
#endif

extern "C" {

void llama_glue_pt(int8_t *in1, int8_t *in2, int8_t *out) {
  for (int i = 0; i < LLAMA_GLUE_PT_BYTES; i++) {
    out[i] = in1[i] ^ in2[i];
  }
}

} // extern "C"
