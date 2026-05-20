//===- yolo_m9_proj_skip_row.cc -------------------------------*- C++ -*-===//
//
// Fused attn/proj 1x1 + cross-scale skip-add (b) for the PSA pipe.
//
// ONNX flow:
//     proj_fp  = (acc * w_scale * in_scale)  [conv on attn_pre_proj]
//     proj_i8  = round(proj_fp / proj_out_scale)        # QL at 2^-5
//     proj_fp' = proj_i8 * proj_out_scale               # DQ
//     b_fp     = b_i8 * b_scale                         # b at 2^-4
//     add_fp   = proj_fp' + b_fp
//     add_i8   = round(add_fp / add_out_scale)          # QL at 2^-4
//
// In integer (proj_out_scale=2^-5, b_scale=2^-4, add_out_scale=2^-4):
//     add_i8 = clip_i8(banker_srs(proj_i8 + 2 * b_i8, 1))
// The "2 * b" comes from b_scale / proj_out_scale = 2 (b is at twice the
// step size, so its integer value contributes twice as much). The "srs 1"
// finishes the requantize back to scale 2^-4.
//
// Operates one (in_w, out_c) row at a time. Weights are OIYXI8O8 (1x1 → 8x8
// blocks). Outputs at scale 2^-4 (the Add output scale).
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

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_proj_skip_row_i8_i8(
    int8_t *in_row,     // (in_w, in_c) attn_pre_proj
    int8_t *b_cache,    // (in_h, in_w, out_c) full-sample b cache
    int8_t *wts,        // (out_c, in_c, 1, 1) OIYXI8O8
    int32_t *bias,      // (out_c,)
    int8_t *out_row,    // (in_w, out_c)
    const int32_t yi,             // which y row of b_cache to read
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift,        // 7 (proj's rs)
    const int32_t skip_shift) {       // 1 (= log2(b_scale/proj_out_scale))
  event0();

  int8_t *b_row = b_cache + yi * input_width * output_channels;

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t binit = bias[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t proj_q = banker_srs(sum, right_shift);
      if (proj_q > I8_MAX) proj_q = I8_MAX;
      if (proj_q < I8_MIN) proj_q = I8_MIN;
      // Cross-scale add: b is at coarser scale (×2 step size) so its
      // integer value contributes twice as much per unit of step.
      int32_t add_q = proj_q + ((int32_t)b_row[x * output_channels + oc] << skip_shift);
      int32_t add_i8 = banker_srs(add_q, skip_shift);
      if (add_i8 > I8_MAX) add_i8 = I8_MAX;
      if (add_i8 < I8_MIN) add_i8 = I8_MIN;
      out_row[x * output_channels + oc] = (int8_t)add_i8;
    }
  }

  event1();
}

} // extern "C"
