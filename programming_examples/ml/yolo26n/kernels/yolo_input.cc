//===- yolo_input.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One row of a letterboxed RGB image to the network's int8 input: each pixel
// byte through the 256-entry table (the graph's /255 and QuantizeLinear),
// and the three channels padded to the group of 8 the convs read.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef YOLO_INPUT_WIDTH
#error "YOLO_INPUT_WIDTH must be defined"
#endif

extern "C" {

void yolo_input_quant(const uint8_t *__restrict rgb,
                      const int8_t *__restrict table, int8_t *__restrict out) {
  event0();
  for (int x = 0; x < YOLO_INPUT_WIDTH; x++) {
    out[x * 8 + 0] = table[rgb[x * 3 + 0]];
    out[x * 8 + 1] = table[rgb[x * 3 + 1]];
    out[x * 8 + 2] = table[rgb[x * 3 + 2]];
    AIE_LOOP_UNROLL_FULL
    for (int c = 3; c < 8; c++)
      out[x * 8 + c] = 0;
  }
  event1();
}

} // extern "C"
