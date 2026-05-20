//===- yolo_c3k2_heavy_inner_pair_cv2_skip.cc ------------------*- C++ -*-===//
//
// Scalar 3x3 stride-1 INT8 conv + SiLU LUT + CROSS-SCALE skip-add.
//
// The cross-scale add must be performed in higher precision before
// rounding — rounding each operand separately and then adding gives the
// wrong result in fractional cases:
//
//     out_i8 = banker_srs(y_i8 * y_mult + cv2silu_i8 * cv2_mult, rsh_add)
//
// All four c3k2_heavy pair adds (m6/m8 × inner_pair_0/1) work out to
// y_mult=1, cv2_mult=2, rsh_add=1 — equivalent to a banker-rounded
// (y/2 + cv2). Args are runtime so different ratios fit without code
// changes.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) +
         ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv2_skip_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *skip_row,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t skip_y_mult,    // y contribution multiplier (pre-rshift)
    const int32_t skip_cv2_mult,  // cv2silu contribution multiplier (pre-rshift)
    const int32_t skip_rsh_add) { // rshift to bring sum back to s_add
  event0();

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t binit = bias[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        for (int kx = 0; kx < kernel_width; kx++) {
          int col = x - 1 + kx;
          if (col < 0 || col >= input_width) continue;
          int in_indx = col * input_channels + ic;

          int w0 = wts[wts_idx_oiyxi8o8(oc, ic, 0, kx, input_channels,
                                        kernel_height, kernel_width)];
          int w1 = wts[wts_idx_oiyxi8o8(oc, ic, 1, kx, input_channels,
                                        kernel_height, kernel_width)];
          int w2 = wts[wts_idx_oiyxi8o8(oc, ic, 2, kx, input_channels,
                                        kernel_height, kernel_width)];

          if (!skip_top) sum += line0[in_indx] * w0;
          sum += line1[in_indx] * w1;
          if (!skip_bot) sum += line2[in_indx] * w2;
        }
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      int32_t cv2silu = silu_lut[s + 128];
      int32_t y = (int32_t)skip_row[x * output_channels + oc];
      // Cross-scale add in higher precision, then banker-round + clip.
      int32_t sum_pre = y * skip_y_mult + cv2silu * skip_cv2_mult;
      int32_t added = banker_srs(sum_pre, skip_rsh_add);
      added = added > I8_MAX ? I8_MAX : (added < I8_MIN ? I8_MIN : added);
      output[x * output_channels + oc] = (int8_t)added;
    }
  }

  event1();
}

} // extern "C"
