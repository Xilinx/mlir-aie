// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Blocked transpose: every SxS block of a DIM_n x DIM_m (rows x columns)
// matrix is transposed in place, so the blocks stay where they are and the
// elements inside each block move. Built on aie::transpose, the AIE API's
// vector-as-matrix transpose (VSHUFFLE underneath); see
// programming_examples/basic/transposes for a design that combines this
// kernel with DMA-level transposes.
//
// -DDIM_m / -DDIM_n give the tile shape; -DBIT_WIDTH (8, 16 or 32, default
// 16) the element width, as the other generic kernels take it. The kernel
// only moves bytes, so the unsigned integer of that width stands in for any
// dtype: the library's bf16 transpose is BIT_WIDTH=16.

#include <aie_api/aie.hpp>
#include <algorithm>
#include <cstdint>

#if !defined(DIM_m) || !defined(DIM_n)
#error Please specify matrix sizes m, n at kernel compile time using e.g., -DDIM_m=32 -DDIM_n=32.
#endif

#ifndef BIT_WIDTH
#define BIT_WIDTH 16
#endif

#if BIT_WIDTH == 8
using T = uint8_t;
#elif BIT_WIDTH == 16
using T = uint16_t;
#elif BIT_WIDTH == 32
using T = uint32_t;
#else
#error BIT_WIDTH must be 8, 16 or 32.
#endif

constexpr unsigned VEC = 1024 / BIT_WIDTH; // elements in one full-width vector
constexpr unsigned OUTER_SIZE = DIM_m * DIM_n;
constexpr unsigned COPY_VEC = std::min<unsigned>(VEC, OUTER_SIZE);
static_assert(OUTER_SIZE % COPY_VEC == 0);

extern "C" {

void copy(T *__restrict in_ptr, T *__restrict out_ptr) {
  for (unsigned i = 0; i < OUTER_SIZE; i += COPY_VEC)
    aie::store_v(out_ptr + i, aie::load_v<COPY_VEC>(in_ptr + i));
}
}

// The matrix is walked in strips of R rows by W columns that fill one vector.
// W spans C = W / S blocks. Two aie::transpose calls do the work: one per row
// turns its (block, column) order into (column, block); one over the strip
// then turns (row, column, block) into (column, block, row), which is the
// strip of the block-transposed matrix, row by row. When a full-width
// vector holds fewer than S rows (32-bit 8x8 blocks), an output row is
// assembled from the S / R strips that cover the block.
template <unsigned S>
static inline void transpose_blocks(const T *__restrict in, T *__restrict out) {
  constexpr unsigned W =
      std::max<unsigned>(S, std::min<unsigned>(DIM_m, VEC / S));
  constexpr unsigned R = std::min<unsigned>(S, VEC / W);
  constexpr unsigned C = W / S;
  constexpr unsigned H = S / R; // strips per block row
  static_assert(DIM_m % W == 0 && DIM_n % S == 0 && S % R == 0);
  static_assert(W * BIT_WIDTH >= 128, "a strip row must fill a 128-bit load");
  static_assert(H == 1 || R * BIT_WIDTH >= 128,
                "chunks must fill a 128-bit vector");

  for (unsigned row = 0; row < DIM_n; row += S) {
    for (unsigned col = 0; col < DIM_m; col += W) {
      aie::vector<T, R * W> strips[H];
      for (unsigned h = 0; h < H; ++h) {
        aie::vector<T, R * W> a;
        for (unsigned i = 0; i < R; ++i) {
          const T *src = in + (row + h * R + i) * DIM_m + col;
          a.insert(i, aie::transpose(aie::load_v<W>(src), C, S));
        }
        strips[h] = aie::transpose(a, R, W);
      }
      for (unsigned q = 0; q < S; ++q) {
        T *dst = out + (row + q) * DIM_m + col;
        if constexpr (H == 1) {
          aie::store_v(dst, strips[0].template extract<W>(q));
        } else {
          aie::vector<T, W> o;
          for (unsigned c = 0; c < C; ++c)
            for (unsigned h = 0; h < H; ++h)
              o.insert(c * H + h, strips[h].template extract<R>(q * C + c));
          aie::store_v(dst, o);
        }
      }
    }
  }
}

extern "C" {

void transpose_4x4(T *__restrict in_ptr, T *__restrict out_ptr) {
  transpose_blocks<4>(in_ptr, out_ptr);
}

void transpose_8x8(T *__restrict in_ptr, T *__restrict out_ptr) {
  transpose_blocks<8>(in_ptr, out_ptr);
}
}
