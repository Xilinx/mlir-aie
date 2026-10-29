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
static_assert(YOLO_INPUT_MUL >= 1 && YOLO_INPUT_MUL <= 127,
              "the multiplier is an int8 lane");

extern "C" {

// The row is quantized in place: the input is the image's alone.
void yolo_input_quant(uint8_t *__restrict rgb, int8_t *__restrict out) {
  event0();
  const aie::rounding_mode saved_rnd =
      aie::swap_rounding(aie::rounding_mode::positive_inf);
  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  constexpr int quads = YOLO_INPUT_WIDTH / 4;
  int8_t *q = reinterpret_cast<int8_t *>(rgb);
  {
    const uint8_t *in = rgb;
    int8_t *at = q;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(YOLO_INPUT_WIDTH * 3 / 32)
    for (int i = 0; i < YOLO_INPUT_WIDTH * 3 / 32; i++) {
      aie::store_v(at, aie::mul(aie::load_v<32>(in), (int8_t)YOLO_INPUT_MUL)
                           .template to_vector<int8>(YOLO_INPUT_SHIFT));
      in += 32;
      at += 32;
    }
  }
  const int8_t *w0 = q;
  const int8_t *w1 = q + 4 * quads;
  const int8_t *w2 = q + 8 * quads;
  uint32_t *dst = reinterpret_cast<uint32_t *>(out);
  const aie::vector<int8, 64> z = aie::zeros<int8, 64>();
  const aie::vector<uint32, 16> zero = aie::zeros<uint32, 16>();
  // Byte k of each word: whole-byte shifts within a word are byte moves
  // across the vector, kept where the lane's byte k is the one wanted.
  const auto k0 = aie::mask<64>::from_uint64(0x1111111111111111ULL);
  const auto k01 = aie::mask<64>::from_uint64(0x3333333333333333ULL);
  const auto k12 = aie::mask<64>::from_uint64(0x6666666666666666ULL);
  const auto k2 = aie::mask<64>::from_uint64(0x4444444444444444ULL);
  const auto k012 = aie::mask<64>::from_uint64(0x7777777777777777ULL);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(quads / 16)
  for (int i = 0; i < quads / 16; i++) {
    const aie::vector<int8, 64> a = aie::load_v<64>(w0);
    const aie::vector<int8, 64> b = aie::load_v<64>(w1);
    const aie::vector<int8, 64> c = aie::load_v<64>(w2);
    w0 += 64;
    w1 += 64;
    w2 += 64;
    // Pixel k of each four, its three bytes low in a word.
    const aie::vector<uint32, 16> p0 =
        aie::select(z, a, k012).template cast_to<uint32>();
    const aie::vector<uint32, 16> p1 =
        aie::select(aie::select(z, aie::shuffle_up_fill(b, z, 1), k12),
                    aie::shuffle_down_fill(a, z, 3), k0)
            .template cast_to<uint32>();
    const aie::vector<uint32, 16> p2 =
        aie::select(aie::select(z, aie::shuffle_up_fill(c, z, 2), k2),
                    aie::shuffle_down_fill(b, z, 2), k01)
            .template cast_to<uint32>();
    const aie::vector<uint32, 16> p3 =
        aie::select(z, aie::shuffle_down_fill(c, z, 1), k012)
            .template cast_to<uint32>();
    const auto [l01, h01] = aie::interleave_zip(p0, p1, 1);
    const auto [l23, h23] = aie::interleave_zip(p2, p3, 1);
    // Sixteen pixels in order per vector, then a zero word after each.
    const auto [x0, x1] = aie::interleave_zip(l01, l23, 2);
    const auto [x2, x3] = aie::interleave_zip(h01, h23, 2);
    const aie::vector<uint32, 16> pixels[4] = {x0, x1, x2, x3};
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 4; k++) {
      const auto [lo, hi] = aie::interleave_zip(pixels[k], zero, 1);
      aie::store_v(dst + 32 * k, lo);
      aie::store_v(dst + 32 * k + 16, hi);
    }
    dst += 128;
  }
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

} // extern "C"
