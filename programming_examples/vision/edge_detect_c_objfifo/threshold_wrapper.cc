//===- threshold_wrapper.cc - Threshold core body ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "aie_objectfifo.h"

extern "C" void thresholdLine(uint8_t *in, uint8_t *out, int32_t lineWidth,
                              int16_t v_thr, int16_t v_max, int8_t v_typ);

extern "C" {

// Uses 2 buffers per OF for ping-pong double-buffering.
// objectfifo_get_buffer() alternates between the two physical buffers
// to match the DMA's buffer rotation.
void threshold_core(uint8_t *in_buf0, uint8_t *in_buf1, uint8_t *out_buf0,
                    uint8_t *out_buf1, int64_t in_acq, int64_t in_rel,
                    int64_t out_acq, int64_t out_rel, int32_t lineWidth) {
  objectfifo_t of_in = {(int32_t)in_acq, (int32_t)in_rel, -1, 1,
                         2, {in_buf0, in_buf1}};
  objectfifo_t of_out = {(int32_t)out_acq, (int32_t)out_rel, -1, 1,
                          2, {out_buf0, out_buf1}};

  int32_t iter = 0;

  while (1) {
    objectfifo_acquire(&of_in);
    objectfifo_acquire(&of_out);

    uint8_t *in = (uint8_t *)objectfifo_get_buffer(&of_in, iter);
    uint8_t *out = (uint8_t *)objectfifo_get_buffer(&of_out, iter);

    thresholdLine(in, out, lineWidth, 10, 255, 0);

    objectfifo_release(&of_in);
    objectfifo_release(&of_out);
    iter++;
  }
}

} // extern "C"
