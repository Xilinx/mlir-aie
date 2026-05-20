//===- yolo_m10_conv2dk1_silu_xy_pool.cc ----------------------*- C++ -*-===//
//
// Fused 1x1 i8 conv 256→1280 + HardSiLU LUT + spatial xy-pool (GAP) for
// the yolo26n-cls head. Weights stream from a memtile in n_splits chunks
// of chunk_oc output channels (e.g., 16 chunks × 80 oc each for the
// 256→1280 default). The kernel is called n_splits × in_h times per
// sample, accumulating the (post-SiLU) spatial sum into a persistent
// i32 buffer (`accum`), then finalizing on the last call by shifting
// the accumulator right by 5 (the combined GAP-divide-by-256 + scale
// change 2^-2 → 2^-5 = >> 5) and clipping to i8.
//
// Persistent accumulator lives in an IRON-allocated Buffer (passed as
// `accum`) since static C++ scratch isn't reliable on AIE2P.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYXI8O8 indexer for 1x1 (chunk_oc, in_c, 1, 1) packed chunks.
static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m10_conv2dk1_silu_xy_pool_i8_i8(
    int8_t *in_row,            // (in_w, in_c) one spatial row
    int8_t *wts_chunk,         // OIYXI8O8 (chunk_oc, in_c, 1, 1)
    int32_t *bias_full,        // (expand_c,)
    int8_t *silu_lut,          // (256,) HardSiLU LUT
    int32_t *accum,            // (expand_c,) persistent i32 accumulator
    int8_t *elem_out,          // (expand_c,) i8 final output (written on last call)
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t expand_c,
    const int32_t in_h,                  // 16
    const int32_t right_shift,
    const int32_t yi,
    const int32_t n_splits,
    const int32_t wi) {
  event0();

  const int32_t chunk_oc = expand_c / n_splits;
  const int32_t oc_offset = wi * chunk_oc;

  // First call of the sample: zero the full accumulator.
  if (yi == 0 && wi == 0) {
    for (int i = 0; i < expand_c; i++) accum[i] = 0;
  }

  // 1x1 conv for chunk_oc output channels; per output sum across input_width
  // spatial positions via SiLU LUT then accumulate.
  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[oc_offset + co];
    int32_t row_sum = 0;
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts_chunk[wts_chunk_idx_1x1(co, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      if (s > I8_MAX) s = I8_MAX;
      if (s < I8_MIN) s = I8_MIN;
      row_sum += (int32_t)silu_lut[s + 128];
    }
    accum[oc_offset + co] += row_sum;
  }

  // Last call (yi=in_h-1, wi=n_splits-1): finalize. ONNX does this as
  // TWO steps (NOT one combined SRS by 5):
  //   (1) AveragePool then QL at 2^-2: pool_i8 = banker_srs(sum, 8)
  //                                              (divide by N=256, banker)
  //   (2) Flatten QL at 2^-5:           flat_i8 = clip(pool_i8 << 3, i8)
  //                                              (scale ratio 2^-2/2^-5 = 8)
  // The two-step path can round differently from one banker_srs(.., 5)
  // for certain values; matters for bit-exact match against ORT.
  if (yi == in_h - 1 && wi == n_splits - 1) {
    for (int i = 0; i < expand_c; i++) {
      int32_t pool_q = banker_srs(accum[i], 8);
      if (pool_q > I8_MAX) pool_q = I8_MAX;
      if (pool_q < I8_MIN) pool_q = I8_MIN;
      int32_t f = pool_q << 3;
      if (f > I8_MAX) f = I8_MAX;
      if (f < I8_MIN) f = I8_MIN;
      elem_out[i] = (int8_t)f;
    }
  }

  event1();
}

} // extern "C"
