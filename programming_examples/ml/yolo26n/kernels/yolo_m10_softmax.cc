//===- yolo_m10_softmax.cc -------------------------------------*- C++ -*-===//
//
// Tiny i8 softmax (n_classes typically = 2 for the yolo26n-cls binary
// classifier head). Uses a precomputed fp32 exp LUT keyed by the i8
// (logit - row_max). Output is i8 at scale 2^-7 (probability * 128).
//
// Same algorithm as yolo_m9_softmax_row but on a flat (n_classes,)
// vector rather than one row of a chunk. No stack scratch needed
// since two-pass is essentially free at n_classes=2.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

static constexpr int32_t I8_MAX_PROB = 127;
static constexpr int32_t I8_MIN_PROB = 0;

extern "C" {

void yolo_m10_softmax_i8_i8(
    int8_t *logits,                // (n_classes,)
    float *exp_lut,                // (256,) exp((idx-128) * 2^in_log2_scale)
    int8_t *out_probs,             // (n_classes,)
    const int32_t n_classes,       // 2
    const int32_t in_log2_scale) { // -3 for m10
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  // Hardcoded for m10 classifier head (n_classes=2). out scale = 2^-7
  // → multiply prob by 128 (computed as int shift to avoid libm).
  (void)n_classes;
  (void)in_log2_scale; // baked into exp_lut at build time
  constexpr int kN = 2;
  constexpr float kOutScale = 128.0f;

  // Pass 1: max.
  int32_t row_max = -128;
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < kN; j++) {
    int32_t v = (int32_t)logits[j];
    if (v > row_max)
      row_max = v;
  }

  // Pass 2: sum exp.
  float sum = 0.0f;
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < kN; j++) {
    int32_t shifted = (int32_t)logits[j] - row_max;
    if (shifted < -128)
      shifted = -128;
    sum += exp_lut[shifted + 128];
  }

  // Pass 3: normalize + quantize. aie::inv(float) is a HW reciprocal
  // (single op) — replaces __divsf3. Fold out_scale into the constant.
  const float scale = kOutScale * aie::inv(sum);
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < kN; j++) {
    int32_t shifted = (int32_t)logits[j] - row_max;
    if (shifted < -128)
      shifted = -128;
    int32_t q = (int32_t)(exp_lut[shifted + 128] * scale + 0.5f);
    if (q > I8_MAX_PROB)
      q = I8_MAX_PROB;
    if (q < I8_MIN_PROB)
      q = I8_MIN_PROB;
    out_probs[j] = (int8_t)q;
  }

  event1();
}

} // extern "C"
