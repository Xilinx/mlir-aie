//===- merge_rows.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef BLOCK
#define BLOCK 64
#endif
#ifndef AUDIO_TOKEN
#define AUDIO_TOKEN -1
#endif
#ifndef IMAGE_TOKEN
#define IMAGE_TOKEN -2
#endif
#ifndef AUDIO_AT
#define AUDIO_AT 0
#endif
#ifndef VISION_AT
#define VISION_AT 0
#endif

namespace {
int audio, image;
}

// The row each of block b's ids takes: its own position, or the next of a
// tower's rows for a placeholder. The counts run over the blocks of a
// sequence and restart at block 0.
extern "C" void merge_rows(int32_t *ids, int32_t *merge, int32_t b) {
  event0();
  if (b == 0)
    audio = image = 0;
  for (int r = 0, i = b * BLOCK; r < BLOCK; r++, i++) {
    int id = ids[r];
    merge[r] = id == AUDIO_TOKEN   ? AUDIO_AT + audio++
               : id == IMAGE_TOKEN ? VISION_AT + image++
                                   : i;
  }
  event1();
}
