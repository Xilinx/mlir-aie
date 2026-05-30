//===- llama_sample.cc --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama sample v0: greedy argmax over int8 logits.
//
//   token_id = argmax_v logits[v]
//
// Inputs from lm_head are int8 logits (per-tensor scale doesn't matter
// for argmax). Output is a single int32 token id. The full design
// (temperature scale + top-k + softmax + multinomial + EOS check) is a
// v1 follow-up requiring a PRNG; greedy is the simplest semantically
// meaningful sample and lets us land the kernel + dataflow shape.
//
// Tie-breaking: first-occurrence (matches numpy argmax default).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef LLAMA_SAMPLE_VOCAB
#define LLAMA_SAMPLE_VOCAB 1024   // tiny first test; production = 128256
#endif

extern "C" {

void llama_sample(int8_t *restrict logits, int32_t *restrict token_id) {
  constexpr int32_t kV = LLAMA_SAMPLE_VOCAB;
  int32_t best_idx = 0;
  int32_t best_val = (int32_t)logits[0];
  for (int32_t v = 1; v < kV; v++) {
    int32_t l = (int32_t)logits[v];
    if (l > best_val) {
      best_val = l;
      best_idx = v;
    }
  }
  token_id[0] = best_idx;
}

} // extern "C"
