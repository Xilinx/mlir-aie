//===- llama_passthrough_f32.cc -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Trivial fp32 chunk copy, used by the memtile write-once/read-Rx probe
// (aie2_memrepeat_probe.py) to validate the repeat_count bridge for M3a.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef LLAMA_PROBE_CHUNK
#define LLAMA_PROBE_CHUNK 64
#endif

extern "C" {

void passthrough_f32(float *restrict in, float *restrict out) {
  for (int i = 0; i < LLAMA_PROBE_CHUNK; i++)
    out[i] = in[i];
}

// Witness copy: read the WHOLE half-buffer (forces the relay DMA of all
// LLAMA_PROBE_HALF elements) but write only a small WITNESS to L1 (first +
// last element of the half), so the consumer's output buffer stays tiny and
// fits L1 even when the half is 256 KB.
#ifndef LLAMA_PROBE_HALF
#define LLAMA_PROBE_HALF LLAMA_PROBE_CHUNK
#endif
void passthrough_f32_witness(float *restrict in, float *restrict out) {
  out[0] = in[0];
  out[1] = in[LLAMA_PROBE_HALF - 1];
}

} // extern "C"
