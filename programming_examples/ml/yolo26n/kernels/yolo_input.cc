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
// The row comes by phase: four pixels are three words, r0 g0 b0 r1 | g1 b1
// r2 g2 | b2 r3 g3 b3, and the DMA sends word 0 of every four pixels, then
// every word 1, then every word 2.
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

static_assert(YOLO_INPUT_WIDTH % 64 == 0, "rows go 64 pixels at a time");

extern "C" {

// The row is quantized in place: the input is the image's alone.
void yolo_input_quant(uint8_t *__restrict rgb, int8_t *__restrict out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::positive_inf);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  constexpr int quads = YOLO_INPUT_WIDTH / 4;
  int8_t *q = reinterpret_cast<int8_t *>(rgb);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(YOLO_INPUT_WIDTH * 3 / 32)
  for (int i = 0; i < YOLO_INPUT_WIDTH * 3; i += 32)
    aie::store_v(q + i,
                 aie::mul(aie::load_v<32>(rgb + i), (int8_t)YOLO_INPUT_MUL)
                     .template to_vector<int8>(YOLO_INPUT_SHIFT));
  const uint32_t *w0 = reinterpret_cast<const uint32_t *>(q);
  const uint32_t *w1 = w0 + quads;
  const uint32_t *w2 = w1 + quads;
  uint32_t *dst = reinterpret_cast<uint32_t *>(out);
  const aie::vector<uint32, 16> zero = aie::zeros<uint32, 16>();
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < quads; i += 16) {
    const aie::vector<uint32, 16> a = aie::load_v<16>(w0 + i);
    const aie::vector<uint32, 16> b = aie::load_v<16>(w1 + i);
    const aie::vector<uint32, 16> c = aie::load_v<16>(w2 + i);
    // Pixel k of each four, its three bytes low in a word.
    const aie::vector<uint32, 16> p0 = aie::bit_and((uint32_t)0xFFFFFF, a);
    const aie::vector<uint32, 16> p1 =
        aie::bit_or(aie::logical_downshift(a, 24),
                    aie::upshift(aie::bit_and((uint32_t)0xFFFF, b), 8));
    const aie::vector<uint32, 16> p2 =
        aie::bit_or(aie::logical_downshift(b, 16),
                    aie::upshift(aie::bit_and((uint32_t)0xFF, c), 16));
    const aie::vector<uint32, 16> p3 = aie::logical_downshift(c, 8);
    const auto [l01, h01] = aie::interleave_zip(p0, p1, 1);
    const auto [l23, h23] = aie::interleave_zip(p2, p3, 1);
    // Sixteen pixels in order per vector, then a zero word after each.
    const auto [x0, x1] = aie::interleave_zip(l01, l23, 2);
    const auto [x2, x3] = aie::interleave_zip(h01, h23, 2);
    uint32_t *o = dst + 8 * i;
    const aie::vector<uint32, 16> pixels[4] = {x0, x1, x2, x3};
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 4; k++) {
      const auto [lo, hi] = aie::interleave_zip(pixels[k], zero, 1);
      aie::store_v(o + 32 * k, lo);
      aie::store_v(o + 32 * k + 16, hi);
    }
  }
  event1();
}

} // extern "C"
