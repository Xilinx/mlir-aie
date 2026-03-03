//===- filter2d_wrapper.cc - 3-phase filter2d core body ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Core body for the filter2d (Sobel edge detection) stage.
// Implements the 3-phase sliding window pattern:
//   1. Preamble: acquire 2 lines, reuse first for top border padding
//   2. Steady state: acquire 3 lines, release 1, slide window
//   3. Postamble: acquire 2 lines, reuse last for bottom border padding
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" void filter2dLine(uint8_t *lineIn0, uint8_t *lineIn1,
                             uint8_t *lineIn2, uint8_t *lineOut,
                             int32_t lineWidth, int16_t *kernel);

extern "C" {

// of_in:  OF_2to3 consumer side (depth 4)
// of_out: OF_3to4 producer side (depth 2, ping-pong)
// kernel_buf: 3x3 Sobel filter coefficients
void filter2d_core(uint8_t *in_buf0, uint8_t *in_buf1, uint8_t *in_buf2,
                   uint8_t *in_buf3, uint8_t *out_buf0, uint8_t *out_buf1,
                   int64_t in_acq, int64_t in_rel, int64_t out_acq,
                   int64_t out_rel, int16_t *kernel_buf, int32_t lineWidth,
                   int32_t height) {
  objectfifo_t of_in = {(int32_t)in_acq, (int32_t)in_rel,
                        -1,      1,
                        4,       {in_buf0, in_buf1, in_buf2, in_buf3}};
  // Output: 2 buffers for ping-pong double-buffering
  objectfifo_t of_out = {(int32_t)out_acq, (int32_t)out_rel,
                         -1,       1,
                         2,        {out_buf0, out_buf1}};

  int32_t in_iter = 0;
  int32_t out_iter = 0;

  while (1) {
    // Phase 1: Preamble - Top border
    objectfifo_acquire_n(&of_in, 2);
    objectfifo_acquire(&of_out);

    uint8_t *line0 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter);
    uint8_t *line1 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter + 1);
    uint8_t *out = (uint8_t *)objectfifo_get_buffer(&of_out, out_iter);

    filter2dLine(line0, line0, line1, out, lineWidth, kernel_buf);
    objectfifo_release(&of_out);
    out_iter++;

    // Phase 2: Steady state - Middle lines
    for (int32_t row = 1; row < height - 1; row++) {
      objectfifo_acquire(&of_in);
      objectfifo_acquire(&of_out);

      line0 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter);
      line1 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter + 1);
      uint8_t *line2 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter + 2);
      out = (uint8_t *)objectfifo_get_buffer(&of_out, out_iter);

      filter2dLine(line0, line1, line2, out, lineWidth, kernel_buf);

      objectfifo_release(&of_in);
      objectfifo_release(&of_out);
      in_iter++;
      out_iter++;
    }

    // Phase 3: Postamble - Bottom border
    objectfifo_acquire(&of_out);

    line0 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter);
    line1 = (uint8_t *)objectfifo_get_buffer(&of_in, in_iter + 1);
    out = (uint8_t *)objectfifo_get_buffer(&of_out, out_iter);

    filter2dLine(line0, line1, line1, out, lineWidth, kernel_buf);

    objectfifo_release_n(&of_in, 2);
    objectfifo_release(&of_out);
    in_iter += 2;
    out_iter++;
  }
}

} // extern "C"
