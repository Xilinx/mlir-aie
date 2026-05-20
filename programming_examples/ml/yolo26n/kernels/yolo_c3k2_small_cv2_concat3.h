//===- yolo_c3k2_small_cv2_concat3.h ----------------------------------*- C++ -*-===//
//
// 1x1 INT8 conv with INT32 bias accumulator-init + SiLU LUT epilogue over
// three concatenated input rows. The three input buffers (top, bot, m0),
// each (in_w, 1, c) where c = three_c/3, are treated as the input channel
// axis of length three_c. Weights OIYXI8O8 with kH=kW=1, in_c=three_c.
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_C3K2_SMALL_CV2_CONCAT3_H
#define _YOLO_C3K2_SMALL_CV2_CONCAT3_H

#include <stdint.h>

extern "C" {

void yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8(
    int8_t *in_top,
    int8_t *in_bot,
    int8_t *in_m0,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t three_c,
    const int32_t output_channels,
    const int32_t right_shift);

} // extern "C"

#endif
