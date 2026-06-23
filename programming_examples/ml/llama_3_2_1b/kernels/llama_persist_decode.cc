//===- llama_persist_decode.cc ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal per-token decode step for the persistent-loop probe. A tied tiny LM:
//   embed[token]  -> int8 row (rescaled, like the real embed gather)
//   lm_head       -> logit[v] = dot(embed_row, table[v])   (tied weights)
//   argmax        -> next token id
// The device loops this T times, feeding token_out back to token_in ON-CHIP
// (no host between tokens) -- proving the full autoregressive feedback loop.
// Tiny V/D so the table is resident; isolates the LOOP from the big-model
// dataflow (the real loop swaps this body for the 16-layer chain + 262MB
// lm_head, with weights host-streamed per token).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef LLAMA_PD_V
#define LLAMA_PD_V 64
#endif
#ifndef LLAMA_PD_D
#define LLAMA_PD_D 64
#endif

static constexpr int kV = LLAMA_PD_V;
static constexpr int kD = LLAMA_PD_D;

extern "C" {

// table: int8[kV * kD] embed weights (row-major).
// token:  int32[1] input token id.
// out:    int32[1] next token id.
//
// Decode step that produces a VARYING autoregressive sequence (so the loop's
// token feedback is genuinely exercised -- a tied embed==lm_head argmax is
// self-favoring and yields a worthless t->t fixed point). next token =
// (sum of embed[token] over D, as uint) mod kV. The whole point is that
// next depends on the CURRENT token's embedding row, so a correct sequence
// only arises if the feedback carries each step's output back as the next
// input. Host mirrors this exactly.
void persist_decode_step(int8_t *restrict table, int32_t *restrict token,
                         int32_t *restrict out) {
  int t = token[0];
  if (t < 0)
    t = 0;
  if (t >= kV)
    t = kV - 1;
  const int8_t *embed_row = table + t * kD;

  uint32_t acc = 0;
  for (int i = 0; i < kD; i++)
    acc += (uint32_t)(uint8_t)embed_row[i]; // sum bytes (unsigned)
  out[0] = (int32_t)(acc % (uint32_t)kV);
}

} // extern "C"
