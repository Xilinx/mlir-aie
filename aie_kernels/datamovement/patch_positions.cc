//===- patch_positions.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef BLOCK
#define BLOCK 256
#endif
#ifndef SIDE
#define SIDE 16
#endif
#ifndef POOL
#define POOL 3
#endif
#ifndef POSITIONS
#define POSITIONS 10240
#endif
#ifndef CORES
#define CORES 16
#endif

namespace {
int n, pad, across, wx, wy, tx, ty;
}

// Block b of the patches in pooling-window order: (x, y) side by side in xy,
// x then POSITIONS + y in position, and the patch's row of a CORES-core
// raster in order. Past the image, or for a size that is not whole windows,
// the tables' padding rows and the raster's zero patch.
extern "C" void patch_positions(int32_t *out, int32_t b, int32_t count,
                                int32_t ho, int32_t wo) {
  event0();
  int32_t *xy = out, *position = out + 2 * BLOCK, *order = out + 4 * BLOCK;
  if (b == 0) {
    unsigned bands = ho > 0 ? (unsigned)ho / SIDE : 0;
    unsigned strips = wo > 0 ? (unsigned)wo / SIDE : 0;
    uint64_t all = (uint64_t)bands * strips;
    bool ok = ho % SIDE == 0 && wo % SIDE == 0 && all > 0 &&
              bands % POOL == 0 && strips % POOL == 0 &&
              all <= (uint64_t)count * BLOCK;
    n = ok ? (int)all : 0;
    pad = (strips / CORES + 1) * CORES;
    across = strips / POOL;
    wx = wy = tx = ty = 0;
  }
  for (int r = 0, i = b * BLOCK; r < BLOCK; r++, i++) {
    if (i < n) {
      int x = POOL * tx + wx, y = POOL * ty + wy;
      xy[2 * r] = position[r] = x;
      xy[2 * r + 1] = y;
      position[BLOCK + r] = POSITIONS + y;
      order[r] = y * pad + x;
      if (++wx < POOL)
        continue;
      wx = 0;
      if (++wy < POOL)
        continue;
      wy = 0;
      if (++tx < across)
        continue;
      tx = 0;
      ty++;
    } else {
      xy[2 * r] = xy[2 * r + 1] = POSITIONS;
      position[r] = position[BLOCK + r] = 2 * POSITIONS;
      order[r] = pad - 1;
    }
  }
  event1();
}
