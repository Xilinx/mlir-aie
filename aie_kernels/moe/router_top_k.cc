//===- router_top_k.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A MoE router's top-k expert ids and softmax weights, from one
// sample_select summary (sample.h) with SAMPLE_K_MAX == the router's
// top_k, so the summary already holds exactly top_k candidates.
// Deterministic, unlike sample_combine: no threshold ambiguity and no
// stochastic draw, just the entries and ties the summary already ranked.

#include <aie_api/aie.hpp>

#include "../sample/exp64.h"
#include "../sample/sample.h"

extern "C" float __subsf3(float, float);

namespace {

float bf16_to_f32(uint32_t bits) {
  return __builtin_bit_cast(float, bits << 16);
}

} // namespace

extern "C" {

// summary: one column's sample_select summary (SAMPLE_SUMMARY_WORDS,
// SAMPLE_K_MAX == the router's top_k); ids: SAMPLE_K_MAX expert ids;
// weights: SAMPLE_K_MAX bf16 softmax weights, ids[i] paired with weights[i].
void router_top_k(const int32_t *summary, int32_t *ids, bfloat16 *weights) {
  event0();
  int32_t n = 0;
  float values[SAMPLE_K_MAX];
  const int32_t entries = summary[SAMPLE_ENTRIES];
  for (int32_t e = 0; e < entries; ++e) {
    ids[n] = summary[SAMPLE_HEADER + 2 * e];
    values[n] = bf16_to_f32(sample_bits(summary[SAMPLE_HEADER + 2 * e + 1]));
    ++n;
  }
  const float tau_value = bf16_to_f32(sample_bits(summary[SAMPLE_TAU]));
  const uint32_t *bitmap = (const uint32_t *)(summary + SAMPLE_BITMAP);
  for (int32_t w = 0; n < SAMPLE_K_MAX; ++w) {
    uint32_t bits = bitmap[w];
    while (bits && n < SAMPLE_K_MAX) {
      const int32_t i = sample_ctz(bits);
      bits &= bits - 1;
      ids[n] = w * 32 + i;
      values[n] = tau_value;
      ++n;
    }
  }
  // The row's largest key is always among the top-k, so the summary's own
  // SAMPLE_MAX_KEY is the softmax max -- no float comparison needed, the
  // same reason sample_combine.cc never compares raw floats either.
  const float max_value = bf16_to_f32(sample_bits(summary[SAMPLE_MAX_KEY]));
  double w[SAMPLE_K_MAX];
  double sum = 0.0;
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i) {
    const float d = __subsf3(values[i], max_value);
    w[i] = exp64((double)d);
    sum += w[i];
  }
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i)
    weights[i] = (bfloat16)(float)(w[i] / sum);
  event1();
}

} // extern "C"
