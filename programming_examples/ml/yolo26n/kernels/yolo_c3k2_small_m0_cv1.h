//===- yolo_c3k2_small_m0_cv1.h ---------------------------------------*- C++ -*-===//
//
// 3x3 stride-1 INT8 conv with INT32 bias accumulator-init and SiLU LUT
// epilogue. One output row per call from 3 input rows. Border arg matches
// conv_stride: 0 = skip top row, 1 = use all 3 rows, 2 = skip bottom row.
//
// Activation layout: (in_w, 1, in_c) int8; output (in_w, 1, out_c) int8.
// Weight layout:     OIYXI8O8 packed int8 with kH=kW=3.
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_C3K2_SMALL_M0_CV1_H
#define _YOLO_C3K2_SMALL_M0_CV1_H

#include <stdint.h>

extern "C" {

void yolo_c3k2_small_m0_cv1_conv2dk3_silu_bias_i8_i8(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t padding);

} // extern "C"

#endif
