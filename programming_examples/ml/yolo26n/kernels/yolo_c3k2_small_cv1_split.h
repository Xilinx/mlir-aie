//===- yolo_c3k2_small_cv1_split.h ------------------------------------*- C++ -*-===//
//
// 1x1 INT8 conv with INT32 bias accumulator-init, SiLU LUT epilogue,
// output split into two equal halves (top = first c channels,
// bot = second c channels).
//
// Activation layout: (in_w, 1, in_c) int8.
// Weight layout:     OIYXI8O8 packed int8.
// Output layouts:    out_top, out_bot each (in_w, 1, twoc/2) int8.
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_C3K2_SMALL_CV1_SPLIT_H
#define _YOLO_C3K2_SMALL_CV1_SPLIT_H

#include <stdint.h>

extern "C" {

void yolo_c3k2_small_cv1_split_silu_bias_i8_i8(
    int8_t *in_row,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift);

} // extern "C"

#endif
