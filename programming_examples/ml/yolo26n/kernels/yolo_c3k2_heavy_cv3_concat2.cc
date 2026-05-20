//===- yolo_c3k2_heavy_cv3_concat2.cc -------------------------*- C++ -*-===//
//
// Scalar 1x1 INT8 conv on two concatenated input rows + SiLU LUT. ic
// indices [0, c') come from in_a (inner pair 1's add output), [c', 2c')
// from in_b (m.0/cv2's split-B branch). Weights packed OIYXI8O8 over
// the full two_cp input axis.
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

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_cv3_concat2_silu_bias_i8_i8)(
    int8_t *in_a,
    int8_t *in_b,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t two_cp,
    const int32_t output_channels,
    const int32_t right_shift) {
  event0();

  const int32_t cp = two_cp / 2;

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t binit = bias[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < two_cp; ic++) {
        int8_t a = (ic < cp) ? in_a[x * cp + ic]
                             : in_b[x * cp + (ic - cp)];
        sum += a * wts[wts_idx_oiyxi8o8_1x1(oc, ic, two_cp)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      output[x * output_channels + oc] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"
