//===- gray2rgba_addweighted_wrapper.cc - Dual-phase core -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Core body for the final stage of edge detect pipeline.
// Two phases per iteration:
//   1. gray2rgba: convert thresholded binary image back to RGBA
//   2. addWeighted: blend edge-detected RGBA with original input (50/50)
//
// Uses a local feedback ObjectFIFO (depth 1) for intermediate data
// between the two phases.
//
// OF_4to5 (in): depth 2, ping-pong
// inOF_L2L1 (in2): depth 2 on consumer side, ping-pong
// OF_local (local): depth 1, single buffer
// outOF_L1L2 (out): depth 2, ping-pong
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" void gray2rgbaLine(uint8_t *in, uint8_t *out, int32_t lineWidth);
extern "C" void addWeightedLine(uint8_t *in1, uint8_t *in2, uint8_t *out,
                                int32_t lineWidthInBytes, int16_t alpha,
                                int16_t beta, int8_t gamma);

extern "C" {

void gray2rgba_addweighted_core(
    uint8_t *in_buf0, uint8_t *in_buf1, uint8_t *in2_buf0, uint8_t *in2_buf1,
    uint8_t *in2_buf2, uint8_t *in2_buf3, uint8_t *local_buf0,
    uint8_t *out_buf0, uint8_t *out_buf1, int64_t in_acq, int64_t in_rel,
    int64_t in2_acq, int64_t in2_rel, int64_t local_p_acq,
    int64_t local_p_rel, int64_t local_c_acq, int64_t local_c_rel,
    int64_t out_acq, int64_t out_rel, int32_t lineWidth,
    int32_t lineWidthInBytes) {
  // OF_4to5: depth 2, ping-pong
  objectfifo_t of_in = {(int32_t)in_acq, (int32_t)in_rel, -1, 1,
                         2, {in_buf0, in_buf1}};
  // inOF_L2L1: consumer depth 4, must match DMA BD chain
  objectfifo_t of_in2 = {(int32_t)in2_acq, (int32_t)in2_rel, -1, 1,
                          4, {in2_buf0, in2_buf1, in2_buf2, in2_buf3}};
  // OF_local: depth 1, single buffer
  objectfifo_t of_local_p = {(int32_t)local_p_acq, (int32_t)local_p_rel, -1, 1,
                             1, {local_buf0}};
  objectfifo_t of_local_c = {(int32_t)local_c_acq, (int32_t)local_c_rel, -1, 1,
                             1, {local_buf0}};
  // outOF_L1L2: depth 2, ping-pong
  objectfifo_t of_out = {(int32_t)out_acq, (int32_t)out_rel, -1, 1,
                          2, {out_buf0, out_buf1}};

  int32_t iter = 0;

  while (1) {
    // Phase 1: gray2rgba -- convert binary edges to RGBA into local buffer
    objectfifo_acquire(&of_in);
    objectfifo_acquire(&of_local_p);

    uint8_t *in = (uint8_t *)objectfifo_get_buffer(&of_in, iter);

    gray2rgbaLine(in, local_buf0, lineWidth);

    objectfifo_release(&of_in);
    objectfifo_release(&of_local_p);

    // Phase 2: addWeighted -- blend edge RGBA with original input RGBA
    objectfifo_acquire(&of_local_c);
    objectfifo_acquire(&of_in2);
    objectfifo_acquire(&of_out);

    uint8_t *in2 = (uint8_t *)objectfifo_get_buffer(&of_in2, iter);
    uint8_t *out = (uint8_t *)objectfifo_get_buffer(&of_out, iter);

    addWeightedLine(local_buf0, in2, out, lineWidthInBytes,
                    16384,  // alpha (0.5 in Q15)
                    16384,  // beta (0.5 in Q15)
                    0       // gamma
    );

    objectfifo_release(&of_local_c);
    objectfifo_release(&of_in2);
    objectfifo_release(&of_out);
    iter++;
  }
}

} // extern "C"
