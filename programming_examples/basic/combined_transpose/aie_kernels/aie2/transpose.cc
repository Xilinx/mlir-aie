// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// This kernel uses the higher-level AIE API to perform a transpsoe.
// This higher-level API uses VSHUFFLE intrinsics internally.
// See programming_examples/basic/shuffle_transpsoe for a transpose
// example at that level.

#include <aie_api/aie.hpp>
#include <cstdint>
#include <cassert>

#if !defined(DIM_m) || !defined(DIM_n) || !defined(DIM_r) || !defined(DIM_s)
#error Please specify matrix sizes m, n, r, s at kernel compile time using e.g., -DDIM_m=32 -DDIM_n=32 -DDIM_r=16 -DDIM_s=8.
#endif

#define DTYPE uint8_t

static_assert(DIM_m % DIM_r == 0);
static_assert(DIM_n % DIM_s == 0);

constexpr size_t INNER_SIZE = DIM_r * DIM_s;
constexpr size_t OUTER_SIZE = DIM_m * DIM_n;

// Constraints for DIM_m and DIM_n
// Judging from aie_api/detail/aie2/transpose.hpp, aie::transpose supports the
// following dimensions:
//  - row/col count must be a power of two
//  - for  8-bit data types, total matrix size must be 128, 64, 32 or 16 bytes
//  - for 16-bit data types, total matrix size must be      64, 32, 16 or 8
//  bytes
//  - for 32-bit data types, total matrix size must be          32, 16, 8 or 4
//  bytes
//  - for 64-bit data types, total matrix size must be              16, 8 or 4
//  bytes
// Let's enforce these constraints here, because the AIE API compilation errors
// are not very helpful; if you end up here because of a failing assertion, it's
// because the AIE API does not support a transpose of the size you requested.
#define IS_POWER_OF_TWO(x) ((x > 0) && ((x & (x - 1)) == 0))
#define IMPLIES(a, b) (!(a) || ((a) && (b))) // a implies b
static_assert(IS_POWER_OF_TWO(DIM_m) && IS_POWER_OF_TWO(DIM_n) &&
              "m and n must be powers of two");
static_assert(IMPLIES(sizeof(DTYPE) == 8,
                      INNER_SIZE == 128 || INNER_SIZE == 64 || INNER_SIZE == 32 || INNER_SIZE == 16));
static_assert(IMPLIES(sizeof(DTYPE) == 16,
                      INNER_SIZE == 64 || INNER_SIZE == 32 || INNER_SIZE == 16 || INNER_SIZE == 8));
static_assert(IMPLIES(sizeof(DTYPE) == 32,
                      INNER_SIZE == 32 || INNER_SIZE == 16 || INNER_SIZE == 8 || INNER_SIZE == 4));
static_assert(IMPLIES(sizeof(DTYPE) == 64,
                      INNER_SIZE == 16 || INNER_SIZE == 8 || INNER_SIZE == 4));

extern "C" {

void transpose(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  // in and out may not alias, i.e. you cannot transpose in-place with this kernel

  // first, each r*s-sized subtile gets transposed within itself
  for (unsigned row = 0; row < DIM_m; row += DIM_r) {
    for (unsigned col = 0; col < DIM_n; col += DIM_s) {
      aie::vector<DTYPE, INNER_SIZE> in = aie::load_v<INNER_SIZE>(in_ptr);
      aie::vector<DTYPE, INNER_SIZE> transposed = aie::transpose(in, DIM_r, DIM_s);
      aie::store_v(out_ptr, transposed);
    }
  }

  // with r*s-sized subtiles transposed, we must interleave adjacent tiles in the same column;
  // in the first step, this creates (2r)*s-sized tiles which are transposed, then we
  // interleave (2r)*s-sized tiles with each other to get (4r)*s-sized tiles that are transposed,
  // etc., until the whole m*n-sized tile is transposed completely.
  unsigned cur_r = DIM_r;
  unsigned cur_s = DIM_s;
  for (unsigned tile_rows_remaining = DIM_m / DIM_r; tile_rows_remaining > 0; tile_rows_remaining /= 2) {
    unsigned tile_cols_remaining = DIM_n / cur_s;
    for (unsigned tile_row = 0; tile_row < tile_rows_remaining; tile_row++) {
      for (unsigned tile_col = 0; tile_col < tile_cols_remaining; tile_col++) {
        unsigned row = tile_row * cur_r;
        unsigned col = tile_col * cur_s;
        DTYPE *upper_ptr = in_ptr + row * DIM_n + col;
        DTYPE *lower_ptr = in_ptr + (row + 1) * DIM_n + col;

        aie::vector<DTYPE, INNER_SIZE> upper = aie::load_v<INNER_SIZE>(upper_ptr);
        aie::vector<DTYPE, INNER_SIZE> lower = aie::load_v<INNER_SIZE>(lower_ptr);
        std::pair<aie::vector<DTYPE, INNER_SIZE>, aie::vector<DTYPE, INNER_SIZE>> transposed = aie::interleave_zip(upper, lower, cur_r);
        auto [left, right] = transposed;
        aie::store_v(upper_ptr, left);
        aie::store_v(lower_ptr, right);
      }
    }

    // We merged two tiles across rows, so the resulting tile is twice as tall.
    cur_r *= 2;
    // Since we have limited vector size capacity, we divide the number of columns (s) in a tile
    // by two to retain the overall vector size. Conceptually, you can also picture the tiles
    // growing as we transpose more and more and ignore this reduction in s.
    cur_s /= 2;
  }

}
}
