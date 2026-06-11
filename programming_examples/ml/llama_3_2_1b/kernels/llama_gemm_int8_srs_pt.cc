//===- llama_gemm_int8_srs_pt.cc ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 1 dataflow stub for llama_gemm_int8_srs. Same fifo count as the
// real kernel will have (2 in: act + weight blob; 1 out), but copies the
// first M*N bytes of `act` into `out` and ignores `w_blob` entirely.
// Used only to validate that two input ObjectFifos converge on this tile
// and the output ObjectFifo round-trips to DRAM correctly.
//
// In the real kernel, `w_blob` is a packed payload of (weights || bias ||
// scale) delivered per call via StaticWeightStream, matching the
// cautious-eureka aie2_llama_iron design (one DRAM stream per per-call
// constant set). The CT only has 2 input DMA channels so packed payloads
// are the standard pattern.
//
// Shapes are pinned compile-time at M=8, K=64, N=64 (so M*N == M*K and
// passthrough is identity-shaped).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#ifndef LLAMA_GEMM_PT_M
#define LLAMA_GEMM_PT_M 8
#endif
#ifndef LLAMA_GEMM_PT_K
#define LLAMA_GEMM_PT_K 64
#endif
#ifndef LLAMA_GEMM_PT_N
#define LLAMA_GEMM_PT_N 64
#endif

extern "C" {

void llama_gemm_int8_srs_pt(
    int8_t *act,    // (M, K)
    int8_t *w_blob, // weights+bias+scale, ignored by stub
    int8_t *out) {  // (M, N)
  (void)w_blob;

  constexpr int kOutBytes = LLAMA_GEMM_PT_M * LLAMA_GEMM_PT_N;
  for (int i = 0; i < kOutBytes; i++) {
    out[i] = act[i];
  }
}

} // extern "C"
