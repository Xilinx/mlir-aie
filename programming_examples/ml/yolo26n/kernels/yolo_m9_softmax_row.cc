//===- yolo_m9_softmax_row.cc --------------------------------*- C++ -*-===//
//
// Scalar i8-domain row-wise softmax for the PSA attention. Reads one row
// of i8 scores from a (chunk_rows, N) chunk at chunk_row, computes the
// standard subtract-max → exp → normalize → quantize pipeline.
//
// peano's aie2p libc++ does not provide expf, so the exp is precomputed
// into a 256-entry fp32 LUT keyed by (scores[j] - row_max + 128). The
// LUT is built host-side (scripts/m9_stage.py) from the softmax layer's
// in_log2_scale: LUT[idx] = exp((idx - 128) * 2^in_log2_scale). Output
// is i8 at scale 2^out_log2_scale, so we multiply probabilities by
// 2^-out_log2_scale (=128 for m9) before rounding.
//
// Operates in-place: reads chunk_io[chunk_row, :] and writes the same row.
// Stack scratch: one (N,) fp32 array — 1KB at N=256.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = 0;       // softmax probs in [0, 1] → i8 in [0, 127]

extern "C" {

void yolo_m9_softmax_row_i8_i8(
    int8_t *chunk_io,           // (chunk_rows, N) shared chunk; we touch row chunk_row
    float *exp_lut,             // (256,) fp32 LUT: exp((idx-128) * 2^in_log2_scale)
    const int32_t chunk_row,
    const int32_t N,            // 256
    const int32_t out_log2_scale) {  // -7 for m9 attn softmax (out = prob * 128)
  event0();

  int8_t *row = chunk_io + chunk_row * N;
#ifdef M9_SM_DEBUG_TRIVIAL
  // Bisect: just write a fixed constant to see if the kernel is reached
  // at all (vs hanging during fp32 work).
  for (int j = 0; j < N; j++) row[j] = (int8_t)42;
  event1();
  return;
#endif
#ifdef M9_SM_DEBUG_MAX_ONLY
  // Bisect: only the i8 max scan, write max to every cell. No fp ops.
  int32_t row_max_dbg = -128;
  for (int j = 0; j < N; j++) {
    int32_t v = (int32_t)row[j];
    if (v > row_max_dbg) row_max_dbg = v;
  }
  for (int j = 0; j < N; j++) row[j] = (int8_t)row_max_dbg;
  event1();
  return;
#endif
#ifdef M9_SM_DEBUG_LUT_ONLY
  // Bisect: do max + LUT lookup (one fp32 load per j) but no fp add/mul.
  // Write floor(lut_val * 127) to each cell — single fp mul + cast.
  int32_t row_max2 = -128;
  for (int j = 0; j < N; j++) {
    int32_t v = (int32_t)row[j];
    if (v > row_max2) row_max2 = v;
  }
  for (int j = 0; j < N; j++) {
    int32_t shifted = (int32_t)row[j] - row_max2;
    if (shifted < -128) shifted = -128;
    float e = exp_lut[shifted + 128];
    int32_t q = (int32_t)(e * 127.0f);
    if (q > 127) q = 127;
    if (q < 0) q = 0;
    row[j] = (int8_t)q;
  }
  event1();
  return;
#endif

  // 2^-out_log2_scale via integer shift (avoids libm). For out_log2_scale=-7
  // this is 1<<7 = 128.0f. Assumes out_log2_scale <= 0 (always true here).
  const int32_t out_scale_int = 1 << (-out_log2_scale);
  const float out_scale = (float)out_scale_int;

  // Pass 1: row max.
  int32_t row_max = -128;
  for (int j = 0; j < N; j++) {
    int32_t v = (int32_t)row[j];
    if (v > row_max) row_max = v;
  }

  // Pass 2: accumulate sum (re-fetch exp values from LUT on pass 3 rather
  // than store them — avoids a 1KB stack array which appears to blow the
  // per-tile stack on AIE2P scalar).
  float sum = 0.0f;
  for (int j = 0; j < N; j++) {
    int32_t shifted = (int32_t)row[j] - row_max;   // in [-255, 0]
    if (shifted < -128) shifted = -128;             // clip into LUT range
    sum += exp_lut[shifted + 128];
  }

  // Pass 3: normalize and quantize (re-lookup exp).
  const float inv_sum = 1.0f / sum;
  for (int j = 0; j < N; j++) {
    int32_t shifted = (int32_t)row[j] - row_max;
    if (shifted < -128) shifted = -128;
    float e = exp_lut[shifted + 128];
    float p = e * inv_sum;
    int32_t q = (int32_t)(p * out_scale + 0.5f);
    if (q > I8_MAX) q = I8_MAX;
    if (q < I8_MIN) q = I8_MIN;
    row[j] = (int8_t)q;
  }

  event1();
}

} // extern "C"
