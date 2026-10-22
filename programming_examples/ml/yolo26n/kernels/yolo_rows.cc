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
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < YOLO_COPY_BYTES / 32; i++)
    aie::store_v(out + i * 32, aie::load_v<32>(in + i * 32));
  event1();
}
#endif

#ifdef YOLO_UPSAMPLE_WIDTH
void YOLO_ROWS_SYMBOL(const int8_t *__restrict in, int8_t *__restrict out) {
  event0();
  const uint64_t *src = reinterpret_cast<const uint64_t *>(in);
  uint64_t *dst = reinterpret_cast<uint64_t *>(out);
  for (int g = 0; g < YOLO_UPSAMPLE_GROUPS; g++)
    for (int x = 0; x < YOLO_UPSAMPLE_WIDTH; x++) {
      const uint64_t pixel = src[g * YOLO_UPSAMPLE_WIDTH + x];
      dst[(g * YOLO_UPSAMPLE_WIDTH + x) * 2] = pixel;
      dst[(g * YOLO_UPSAMPLE_WIDTH + x) * 2 + 1] = pixel;
    }
  event1();
}
#endif

} // extern "C"
