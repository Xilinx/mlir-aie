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

static constexpr int32_t I8_MAX_PROB = 127;
static constexpr int32_t I8_MIN_PROB = 0;

extern "C" {

void yolo_m10_softmax_i8_i8(
    int8_t *logits,            // (n_classes,)
    float *exp_lut,            // (256,) exp((idx-128) * 2^in_log2_scale)
    int8_t *out_probs,         // (n_classes,)
    const int32_t n_classes,    // 2
    const int32_t in_log2_scale) {  // -3 for m10
  event0();

  // Output scale = 2^-7 → multiply prob by 2^7 = 128.
  // (Computed via int shift to avoid libm calls.)
  const int32_t out_scale_int = 1 << 7;
  const float out_scale = (float)out_scale_int;
  (void)in_log2_scale;  // baked into exp_lut at build time

  // Pass 1: max.
  int32_t row_max = -128;
  for (int j = 0; j < n_classes; j++) {
    int32_t v = (int32_t)logits[j];
    if (v > row_max) row_max = v;
  }

  // Pass 2: sum exp.
  float sum = 0.0f;
  for (int j = 0; j < n_classes; j++) {
    int32_t shifted = (int32_t)logits[j] - row_max;
    if (shifted < -128) shifted = -128;
    sum += exp_lut[shifted + 128];
  }

  // Pass 3: normalize + quantize (probability * 128, clipped to [0, 127]).
  const float inv_sum = 1.0f / sum;
  for (int j = 0; j < n_classes; j++) {
    int32_t shifted = (int32_t)logits[j] - row_max;
    if (shifted < -128) shifted = -128;
    float p = exp_lut[shifted + 128] * inv_sum;
    int32_t q = (int32_t)(p * out_scale + 0.5f);
    if (q > I8_MAX_PROB) q = I8_MAX_PROB;
    if (q < I8_MIN_PROB) q = I8_MIN_PROB;
    out_probs[j] = (int8_t)q;
  }

  event1();
}

} // extern "C"
