//===- yolo_c3k2_heavy_m_0_split.cc ---------------------------*- C++ -*-===//
//
// Scalar 1x1 INT8 conv with two parallel-branch outputs (same input, two
// independent weight sets / biases / SiLU LUTs / right-shifts). Used by
// c3k2_heavy's m.0/cv1 + m.0/cv2 split: one input row produces two output
// rows on the same compute tile.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Per-block symbol mangling.
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

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8)(
    int8_t *in_row,
    int8_t *wts_a, int32_t *bias_a, int8_t *silu_lut_a,
    int8_t *wts_b, int32_t *bias_b, int8_t *silu_lut_b,
    int8_t *out_a, int8_t *out_b,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels_a,
    const int32_t output_channels_b,
    const int32_t right_shift_a,
    const int32_t right_shift_b) {
  event0();

  // Branch A
  for (int oc = 0; oc < output_channels_a; oc++) {
    const int32_t binit = bias_a[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts_a[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift_a);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      out_a[x * output_channels_a + oc] = silu_lut_a[s + 128];
    }
  }

  // Branch B
  for (int oc = 0; oc < output_channels_b; oc++) {
    const int32_t binit = bias_b[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts_b[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift_b);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      out_b[x * output_channels_b + oc] = silu_lut_b[s + 128];
    }
  }

  event1();
}

} // extern "C"
