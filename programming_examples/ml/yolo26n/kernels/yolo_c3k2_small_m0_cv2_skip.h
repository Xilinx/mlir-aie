//===- yolo_c3k2_small_m0_cv2_skip.h ----------------------------------*- C++ -*-===//
//
// 3x3 stride-1 INT8 conv with INT32 bias accumulator-init, SiLU LUT
// epilogue, then int8-saturating add of skip_row. skip_scale is reserved
// for future cross-scale rescaling — unused for m2/m4 since all chain
// scales match (cv1 silu_out = m.0/Add output scale = Concat scale).
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_C3K2_SMALL_M0_CV2_SKIP_H
#define _YOLO_C3K2_SMALL_M0_CV2_SKIP_H

#include <stdint.h>

extern "C" {

void yolo_c3k2_small_m0_cv2_skip_silu_bias_i8_i8(
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
    const int32_t skip_scale);

} // extern "C"

#endif
