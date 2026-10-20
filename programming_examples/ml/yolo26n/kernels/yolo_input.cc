//===- yolo_input.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One row of a letterboxed RGB image to the network's int8 input: each pixel
// byte through the graph's /255 and QuantizeLinear, which the design proves
// is (byte * YOLO_INPUT_MUL) >> YOLO_INPUT_SHIFT rounded half up and
// saturated, and the three channels padded to the group of 8 the convs read.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#if !defined(YOLO_INPUT_WIDTH) || !defined(YOLO_INPUT_MUL) ||                  \
    !defined(YOLO_INPUT_SHIFT)
#error "YOLO_INPUT_WIDTH, YOLO_INPUT_MUL and YOLO_INPUT_SHIFT must be defined"
#endif

static_assert(YOLO_INPUT_WIDTH % 32 == 0, "rows go 32 pixels at a time");
static_assert(YOLO_INPUT_MUL >= 1 && YOLO_INPUT_MUL <= 127,
              "the multiplier is an int8 lane");

extern "C" {

void yolo_input_quant(const uint8_t *__restrict rgb, int8_t *__restrict out) {
  event0();
  const aie::rounding_mode saved_rnd =
      aie::swap_rounding(aie::rounding_mode::positive_inf);
  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  alignas(32) int8_t q[YOLO_INPUT_WIDTH * 3];
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < YOLO_INPUT_WIDTH * 3; i += 32)
    aie::store_v(q + i,
                 aie::mul(aie::load_v<32>(rgb + i), (int8_t)YOLO_INPUT_MUL)
                     .template to_vector<int8>(YOLO_INPUT_SHIFT));
  // Four pixels are three words: r0 g0 b0 r1 | g1 b1 r2 g2 | b2 r3 g3 b3.
  const uint32_t *w = reinterpret_cast<const uint32_t *>(q);
  uint64_t *dst = reinterpret_cast<uint64_t *>(out);
  AIE_PREPARE_FOR_PIPELINING
  for (int p = 0; p < YOLO_INPUT_WIDTH / 4; p++) {
    const uint32_t w0 = w[3 * p], w1 = w[3 * p + 1], w2 = w[3 * p + 2];
    dst[4 * p] = w0 & 0xFFFFFF;
    dst[4 * p + 1] = w0 >> 24 | (w1 & 0xFFFF) << 8;
    dst[4 * p + 2] = w1 >> 16 | (w2 & 0xFF) << 16;
    dst[4 * p + 3] = w2 >> 8;
  }
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

} // extern "C"
