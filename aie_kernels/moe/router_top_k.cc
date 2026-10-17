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

// A float cast to bfloat16 rounds in the core's mode, floor unless a design
// sets it; this rounds to nearest even whatever the mode. weights are finite.
bfloat16 f32_to_bf16(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint16_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  return __builtin_bit_cast(bfloat16, rounded);
}

} // namespace

extern "C" {

// summary: one column's sample_select summary (SAMPLE_SUMMARY_WORDS,
// SAMPLE_K_MAX == the router's top_k); ids: SAMPLE_K_MAX expert ids;
// weights: 2 * SAMPLE_K_MAX bf16 words, the softmax weight for ids[i] at
// weights[2 * i], a zeroed pad word at weights[2 * i + 1] -- every real
// weight then sits at a 4-byte-aligned bf16 index, which a 1-element
// Copy slice of it needs (the shim's scratch-argument addressing drops
// low bits below 4 bytes).
void router_top_k(const int32_t *summary, int32_t *ids, bfloat16 *weights) {
  event0();
  int32_t n = 0;
  float values[SAMPLE_K_MAX];
  const int32_t entries = summary[SAMPLE_ENTRIES];
  for (int32_t e = 0; e < entries; ++e) {
    ids[n] = summary[SAMPLE_HEADER + 2 * e];
    values[n] = sample_value(summary[SAMPLE_HEADER + 2 * e + 1]);
    ++n;
  }
  const float tau_value = sample_value(summary[SAMPLE_TAU]);
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
  const float max_value = sample_value(summary[SAMPLE_MAX_KEY]);
  double w[SAMPLE_K_MAX];
  double sum = 0.0;
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i) {
    const float d = __subsf3(values[i], max_value);
    w[i] = exp64((double)d);
    sum += w[i];
  }
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i) {
    weights[2 * i] = f32_to_bf16((float)(w[i] / sum));
    weights[2 * i + 1] = (bfloat16)0.0f;
  }
  event1();
}

} // extern "C"
