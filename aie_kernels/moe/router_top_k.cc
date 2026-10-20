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

#include "../common/exp_f32.h"
#include "../sample/exp64.h"
#include "../sample/sample.h"

extern "C" float __subsf3(float, float);
extern "C" float __addsf3(float, float);
extern "C" float __divsf3(float, float);

namespace {

// A float cast to bfloat16 rounds in the core's mode, floor unless a design
// sets it; this rounds to nearest even whatever the mode. weights are finite.
bfloat16 f32_to_bf16(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint16_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  return __builtin_bit_cast(bfloat16, rounded);
}

// How far, in float32 ulps, a fast weight must sit from a bf16 rounding
// midpoint to round as the float64 path does. exp_f32's error is at most
// 9.41 * 2^-24 over every float32 in [-87, 0] (exhaustive NPU sweeps: 9.41
// on npu2, 9.39 on npu1), so a weight is within (2 * 9.41 + K + 1) * 2^-24
// of the float64 quotient: exp twice, K - 1 adds and the divide. One ulp
// more covers the float64 path's (float) cast, and the margin doubles that.
#define ROUTER_MARGIN_ULPS (2 * (SAMPLE_K_MAX + 21))

// The weights in float32: d exactly as the float64 path forms it, exp_f32 on
// the vector unit, then IEEE sums and divides. True only when every weight
// sits farther than ROUTER_MARGIN_ULPS from a bf16 midpoint, so it rounds to
// the float64 path's word; otherwise weights is untouched.
bool fast_weights(const float *d, bfloat16 *weights) {
  // At 16 lanes AIE2 needs half the frame and less program memory than at 32.
  constexpr int W = AIE_ARCH_AIE2 ? 16 : 32;
  constexpr int K = SAMPLE_K_MAX;
  alignas(64) float e[(K + W - 1) / W * W];
  for (int i = 0; i < K; ++i)
    if (!(d[i] >= -87.0f))
      return false;
  const aie::rounding_mode saved =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int c = 0; c < K; c += W) {
    alignas(64) float lane[W] = {};
    for (int i = c; i < K && i < c + W; ++i)
      lane[i - c] = d[i];
    aie::store_v(e + c,
                 exp_f32<W>(aie::accum<accfloat, W>(aie::load_v<W>(lane)))
                     .to_vector<float>());
  }
  aie::set_rounding(saved);
  float sum = e[0];
  for (int i = 1; i < K; ++i)
    sum = __addsf3(sum, e[i]);
  uint32_t bits[K];
  for (int i = 0; i < K; ++i) {
    bits[i] = __builtin_bit_cast(uint32_t, __divsf3(e[i], sum));
    const uint32_t low = bits[i] & 0xffffu;
    const uint32_t from_mid = low >= 0x8000u ? low - 0x8000u : 0x8000u - low;
    if ((bits[i] >> 23) < 27 || from_mid <= ROUTER_MARGIN_ULPS)
      return false;
  }
  for (int i = 0; i < K; ++i) {
    const uint16_t rounded = (bits[i] + 0x7fffu + ((bits[i] >> 16) & 1u)) >> 16;
    weights[2 * i] = __builtin_bit_cast(bfloat16, rounded);
    weights[2 * i + 1] = (bfloat16)0.0f;
  }
  return true;
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
  float d[SAMPLE_K_MAX];
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i)
    d[i] = __subsf3(values[i], max_value);
  if (fast_weights(d, weights)) {
    event1();
    return;
  }
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i) {
    w[i] = exp64((double)d[i]);
    sum += w[i];
  }
  for (int32_t i = 0; i < SAMPLE_K_MAX; ++i) {
    weights[2 * i] = f32_to_bf16((float)(w[i] / sum));
    weights[2 * i + 1] = (bfloat16)0.0f;
  }
  event1();
}

} // extern "C"
