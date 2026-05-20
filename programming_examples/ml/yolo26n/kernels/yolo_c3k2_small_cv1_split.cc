//===- yolo_c3k2_small_cv1_split.cc -----------------------------------*- C++ -*-===//
//
// Scalar 1x1 INT8 conv with OIYXI8O8 weight layout. Linear-only; bias
// initializes the accumulator. Output is split channel-wise into two halves
// written to separate buffers (top = channels [0, c), bot = [c, 2c)).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Per-block symbol mangling. IRON's Python Kernel ties the MLIR func
// sym_name to the .o exported symbol, so a shared .o serving two blocks
// with different shape signatures triggers `redefinition of symbol`.
// Compile this .cc N times with -DKERNEL_SUFFIX=_mN to emit N .o files,
// each exporting a block-specific symbol the builder can wire up by name.
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

// OIYXI8O8 indexer for 1x1 kernel (kH=kW=1).
static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv1_split_silu_bias_i8_i8)(
    int8_t *in_row,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift) {
  event0();

  const int32_t c = output_channels >> 1;

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t bias_init = bias[oc];
    int8_t *dst = (oc < c) ? out_top : out_bot;
    const int32_t dst_oc = (oc < c) ? oc : (oc - c);

    for (int x = 0; x < input_width; x++) {
      int32_t sum = bias_init;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      dst[x * c + dst_oc] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"
