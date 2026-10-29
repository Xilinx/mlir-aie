//===- yolo_rows.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Row plumbing for the yolo26n design, one entry point per build, named by
// YOLO_ROWS_SYMBOL: YOLO_COPY_BYTES copies one part of a concatenated row;
// YOLO_UPSAMPLE_WIDTH writes every 8-byte pixel of a [C/8][W][8] row of
// YOLO_UPSAMPLE_GROUPS channel groups twice (a nearest 2x upsample).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

extern "C" {

#ifdef YOLO_COPY_BYTES
static_assert(YOLO_COPY_BYTES % 32 == 0, "copies go 32 bytes at a time");

void YOLO_ROWS_SYMBOL(const int8_t *__restrict in, int8_t *__restrict out) {
  event0();
  // Parts and ring slots sit 32-byte aligned, not always 64.
  constexpr int vectors = YOLO_COPY_BYTES / 32;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(vectors)
  for (int i = 0; i < vectors; i++) {
    aie::store_v(out, aie::load_v<32>(in));
    in += 32;
    out += 32;
  }
  event1();
}
#endif

#ifdef YOLO_UPSAMPLE_WIDTH
static_assert(YOLO_UPSAMPLE_WIDTH * YOLO_UPSAMPLE_GROUPS % 4 == 0,
              "upsamples go 4 pixels at a time");

void YOLO_ROWS_SYMBOL(const int8_t *__restrict in, int8_t *__restrict out) {
  event0();
  constexpr int vectors = YOLO_UPSAMPLE_WIDTH * YOLO_UPSAMPLE_GROUPS / 4;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(vectors)
  for (int i = 0; i < vectors; i++) {
    const aie::vector<int8, 32> pixels = aie::load_v<32>(in);
    const auto [lo, hi] = aie::interleave_zip(pixels, pixels, 8);
    aie::store_v(out, lo);
    aie::store_v(out + 32, hi);
    in += 32;
    out += 64;
  }
  event1();
}
#endif

} // extern "C"
