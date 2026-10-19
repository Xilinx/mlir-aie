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
  const uint8_t *map = reinterpret_cast<const uint8_t *>(table);
  uint64_t *dst = reinterpret_cast<uint64_t *>(out);
  AIE_PREPARE_FOR_PIPELINING
  for (int x = 0; x < YOLO_INPUT_WIDTH; x++)
    dst[x] = (uint64_t)map[rgb[x * 3 + 0]] |
             (uint64_t)map[rgb[x * 3 + 1]] << 8 |
             (uint64_t)map[rgb[x * 3 + 2]] << 16;
  event1();
}

} // extern "C"
