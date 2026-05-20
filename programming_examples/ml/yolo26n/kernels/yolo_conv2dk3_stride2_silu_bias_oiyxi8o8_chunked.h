//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked.h ---------*- C++ -*-===//
//
// Chunked variant of yolo_conv2dk3_stride2_silu_bias_oiyxi8o8: processes
// `oc_count` output channels at offset `oc_offset` out of `output_channels`
// total. Used by m3/m5/m7 where the full weight tensor exceeds the AIE2P
// compute tile's 64KB L1; the IRON builder streams chunks of weights from
// a MemTile via lowlevel_dma.StaticWeightStream.
//
// Per call the kernel writes outputs at output[x*output_channels + oc_full]
// for oc_full in [oc_offset, oc_offset + oc_count). Other output channels
// are untouched (so the same row buffer is read-modify-written across the
// n_splits chunk calls before being released).
//
// Weight chunk layout: OIYXI8O8 packed but for `oc_count` output channels.
// Flat shape: (oc_count/8, in_c/8, kH, kW, 8 [I-inner], 8 [O-inner]).
// Chunk-local oc index: chunk_oc = oc_full - oc_offset.
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_CONV2DK3_STRIDE2_SILU_BIAS_OIYXI8O8_CHUNKED_H
#define _YOLO_CONV2DK3_STRIDE2_SILU_BIAS_OIYXI8O8_CHUNKED_H

#include <stdint.h>

extern "C" {

void yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_i8_i8(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts_chunk,       // weights for oc_count output channels (OIYXI8O8)
    int32_t *bias,           // FULL bias (output_channels), indexed by oc_full
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,  // total in the layer (for output stride)
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t oc_offset,
    const int32_t oc_count);

} // extern "C"

#endif
