//===- maxpool2d.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 5x5 max pool, stride 1, over int8 rows laid out [C/8][W][8]: one output
// row from the five input rows around it. Pixels past the row's ends count
// as -inf (int8 -128 cannot raise a max). A caller at the top or bottom of
// the image passes an edge row again in place of a missing one, which max
// leaves unchanged.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef MAXPOOL_WIDTH
#error "MAXPOOL_WIDTH must be defined"
#endif
#ifndef MAXPOOL_CHANNELS
#error "MAXPOOL_CHANNELS must be defined"
#endif

// Pixels per vector: 8, or 4 when a row of 8k + 4 pixels would leave every
// other channel group off the 64-byte alignment of an 8-pixel load.
template <int M>
static void maxpool5_i8(const int8_t *l0, const int8_t *l1, const int8_t *l2,
                        const int8_t *l3, const int8_t *l4,
                        int8_t *__restrict out) {
  constexpr int W = MAXPOOL_WIDTH;
  constexpr int V = M * 8;
  static_assert(W % M == 0, "the max pool tiles the row by M pixels");
  const aie::vector<int8, V> floor = aie::broadcast<int8, V>(-128);
  event0();
  for (int g = 0; g < MAXPOOL_CHANNELS / 8; g++) {
    const int base = g * W * 8;
    auto column = [&](int x) {
      if (x < 0 || x >= W)
        return floor;
      const int at = base + x * 8;
      aie::vector<int8, V> m =
          aie::max(aie::load_v<V>(l0 + at), aie::load_v<V>(l1 + at));
      m = aie::max(m, aie::load_v<V>(l2 + at));
      m = aie::max(m, aie::load_v<V>(l3 + at));
      return aie::max(m, aie::load_v<V>(l4 + at));
    };
    aie::vector<int8, V> prev = floor;
    aie::vector<int8, V> cur = column(0);
    for (int x = 0; x < W; x += M) {
      aie::vector<int8, V> next = column(x + M);
      aie::vector<int8, V> m =
          aie::max(cur, aie::shuffle_down_fill(prev, cur, V - 16));
      m = aie::max(m, aie::shuffle_down_fill(prev, cur, V - 8));
      m = aie::max(m, aie::shuffle_down_fill(cur, next, 8));
      m = aie::max(m, aie::shuffle_down_fill(cur, next, 16));
      aie::store_v(out + base + x * 8, m);
      prev = cur;
      cur = next;
    }
  }
  event1();
}

extern "C" {

void maxpool2d_5x5_i8(int8_t *l0, int8_t *l1, int8_t *l2, int8_t *l3,
                      int8_t *l4, int8_t *out) {
  maxpool5_i8<MAXPOOL_WIDTH % 8 == 0 ? 8 : 4>(l0, l1, l2, l3, l4, out);
}

} // extern "C"
