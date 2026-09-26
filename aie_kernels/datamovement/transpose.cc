// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Blocked transpose: every SxS block of a DIM_n x DIM_m (rows x columns)
// matrix is transposed in place, so the blocks stay where they are and the
// elements inside each block move. Built on aie::transpose, the AIE API's
// vector-as-matrix transpose (VSHUFFLE underneath), except on AIE2 and AIE2P
// for 16- and 32-bit 8x8 blocks and 8- and 16-bit 4x4 blocks, which call
// VSHUFFLE directly; see
// programming_examples/basic/transposes for a design that combines this
// kernel with DMA-level transposes.
//
// -DDIM_m / -DDIM_n give the tile shape; -DBIT_WIDTH (8, 16 or 32, default
// 16) the element width, as the other generic kernels take it. The kernel
// only moves bytes, so the unsigned integer of that width stands in for any
// dtype: the library's bf16 transpose is BIT_WIDTH=16.

#include "../aie_kernel_utils.h"
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
  event0();
  auto src = aie::begin_restrict_vector<COPY_VEC>(in_ptr);
  auto dst = aie::begin_restrict_vector<COPY_VEC>(out_ptr);
  AIE_LOOP_UNROLL(2)
  for (unsigned i = 0; i < OUTER_SIZE / COPY_VEC; ++i)
    *dst++ = *src++;
  event1();
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
struct Strips {
  static constexpr unsigned W =
      std::max<unsigned>(S, std::min<unsigned>(DIM_m, VEC / S));
  static constexpr unsigned R = std::min<unsigned>(S, VEC / W);
  static constexpr unsigned C = W / S;
  static constexpr unsigned H = S / R; // strips per block row
  static_assert(DIM_m % W == 0 && DIM_n % S == 0 && S % R == 0);
  static_assert(W * BIT_WIDTH >= 128, "a strip row must fill a 128-bit load");
  static_assert(H == 1 || R * BIT_WIDTH >= 128,
                "chunks must fill a 128-bit vector");

  // The strip's own loops all have compile-time trip counts of at most S and
  // index the vectors with the counter, so unrolling them keeps `strips` in
  // registers instead of on the stack.
  static inline void transpose(const T *__restrict in, T *__restrict out,
                               unsigned row, unsigned col) {
    aie::vector<T, R * W> strips[H];
    AIE_LOOP_UNROLL_FULL
    for (unsigned h = 0; h < H; ++h) {
      aie::vector<T, R * W> a;
      AIE_LOOP_UNROLL_FULL
      for (unsigned i = 0; i < R; ++i) {
        const T *src = in + (row + h * R + i) * DIM_m + col;
        a.insert(i, aie::transpose(aie::load_v<W>(src), C, S));
      }
      strips[h] = aie::transpose(a, R, W);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned q = 0; q < S; ++q) {
      T *dst = out + (row + q) * DIM_m + col;
      if constexpr (H == 1) {
        aie::store_v(dst, strips[0].template extract<W>(q));
      } else {
        aie::vector<T, W> o;
        AIE_LOOP_UNROLL_FULL
        for (unsigned c = 0; c < C; ++c) {
          AIE_LOOP_UNROLL_FULL
          for (unsigned h = 0; h < H; ++h)
            o.insert(c * H + h, strips[h].template extract<R>(q * C + c));
        }
        aie::store_v(dst, o);
      }
    }
  }
};

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
// 8x8 blocks of 16- or 32-bit elements from 256-bit rows: two blocks a row at
// 16 bits, one at 32. Each two-register VSHUFFLE moves one bit of the element
// index between the register number and the lane: three stages for 16 bits
// (row bit 1 -> column bit 0, row bit 2 -> block, block -> column bit 2), two
// for 32 bits. AIE2P runs that sequence over 32 columns a call so that the loop
// has enough independent shuffles to fill its schedule.
// 4x4 blocks of 8- or 16-bit elements take 32 columns of four rows, with rows j
// and j + 2 in one register, in three stages. W is the columns a call covers.
// The loops unroll fully so the register arrays stay in registers.
template <unsigned S, unsigned BW>
struct Shuffles {
  static constexpr bool fits = false;
};
template <>
struct Shuffles<4, 8> {
  using U = uint8_t;
  static constexpr unsigned W = 32;
  static constexpr bool fits = DIM_m % W == 0;
  static inline void transpose(const U *__restrict in, U *__restrict out) {
    v64uint8 x[2], y[2], z[2], w[2];
    AIE_LOOP_UNROLL_FULL
    for (unsigned j = 0; j < 2; ++j)
      x[j] = aie::concat(aie::load_v<W>(in + j * DIM_m),
                         aie::load_v<W>(in + (j + 2) * DIM_m));
    y[0] = ::shuffle(x[0], x[1], T8_2x64_lo);
    y[1] = ::shuffle(x[0], x[1], T8_2x64_hi);
    z[0] = ::shuffle(y[0], y[1], T64_2x8_lo);
    z[1] = ::shuffle(y[0], y[1], T64_2x8_hi);
    w[0] = ::shuffle(z[0], z[1], T16_16x4_lo);
    w[1] = ::shuffle(z[0], z[1], T16_16x4_hi);
    AIE_LOOP_UNROLL_FULL
    for (unsigned q = 0; q < 4; ++q)
      aie::store_v(out + q * DIM_m,
                   aie::vector<U, 2 * W>(w[q >> 1]).template extract<W>(q & 1));
  }
};
template <>
struct Shuffles<4, 16> {
  using U = uint16_t;
  static constexpr unsigned W = 32;
  static constexpr bool fits = DIM_m % W == 0;
  static inline void transpose(const U *__restrict in, U *__restrict out) {
    v32uint16 x[2][2], y[2][2], z[2][2], w[2][2];
    AIE_LOOP_UNROLL_FULL
    for (unsigned h = 0; h < 2; ++h)
      AIE_LOOP_UNROLL_FULL
    for (unsigned j = 0; j < 2; ++j)
      x[h][j] =
          aie::concat(aie::load_v<W / 2>(in + j * DIM_m + h * W / 2),
                      aie::load_v<W / 2>(in + (j + 2) * DIM_m + h * W / 2));
    AIE_LOOP_UNROLL_FULL
    for (unsigned j = 0; j < 2; ++j) {
      y[0][j] = ::shuffle(x[0][j], x[1][j], T16_16x4_lo);
      y[1][j] = ::shuffle(x[0][j], x[1][j], T16_16x4_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < 2; ++c) {
      z[c][0] = ::shuffle(y[c][0], y[c][1], T64_8x2_lo);
      z[c][1] = ::shuffle(y[c][0], y[c][1], T64_8x2_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < 2; ++c) {
      w[c][0] = ::shuffle(z[c][0], z[c][1], T16_4x16_lo);
      w[c][1] = ::shuffle(z[c][0], z[c][1], T16_4x16_hi);
    }
    // AIE2 unrolls this loop unasked, into a schedule the pragma would change.
#if AIE_TUNED_AIE2P
    AIE_LOOP_UNROLL_FULL
#endif
    for (unsigned q = 0; q < 4; ++q)
      aie::store_v(out + q * DIM_m, aie::vector<U, W>(w[q >> 1][q & 1]));
  }
};
template <>
struct Shuffles<8, 16> {
  using U = uint16_t;
  static constexpr unsigned V = 16; // columns of one sequence
  static constexpr unsigned W = AIE_TUNED_AIE2P ? 2 * V : V;
  static constexpr bool fits = DIM_m % W == 0;
  static inline void transpose(const U *__restrict in, U *__restrict out) {
    AIE_LOOP_UNROLL_FULL
    for (unsigned v = 0; v < W; v += V)
      sequence(in + v, out + v);
  }
  static inline void sequence(const U *__restrict in, U *__restrict out) {
    v32uint16 x[4], y[2][2], z[2][2], w[2][2];
    AIE_LOOP_UNROLL_FULL
    for (unsigned k = 0; k < 4; ++k)
      x[k] = aie::concat(aie::load_v<V>(in + 2 * k * DIM_m),
                         aie::load_v<V>(in + (2 * k + 1) * DIM_m));
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < 2; ++r) {
      y[r][0] = ::shuffle(x[2 * r], x[2 * r + 1], T16_32x2_lo);
      y[r][1] = ::shuffle(x[2 * r], x[2 * r + 1], T16_32x2_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < 2; ++c) {
      z[c][0] = ::shuffle(y[0][c], y[1][c], T64_8x2_lo);
      z[c][1] = ::shuffle(y[0][c], y[1][c], T64_8x2_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < 2; ++c) {
      w[c][0] = ::shuffle(z[c][0], z[c][1], T16_16x4_lo);
      w[c][1] = ::shuffle(z[c][0], z[c][1], T16_16x4_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned q = 0; q < 8; ++q)
      aie::store_v(out + q * DIM_m, aie::vector<U, 2 * V>(w[q & 1][q >> 2])
                                        .template extract<V>((q >> 1) & 1));
  }
};
template <>
struct Shuffles<8, 32> {
  using U = uint32_t;
  static constexpr unsigned V = 8; // columns of one sequence
  static constexpr unsigned W = AIE_TUNED_AIE2P ? 4 * V : V;
  static constexpr bool fits = DIM_m % W == 0;
  static inline void transpose(const U *__restrict in, U *__restrict out) {
    AIE_LOOP_UNROLL_FULL
    for (unsigned v = 0; v < W; v += V)
      sequence(in + v, out + v);
  }
  static inline void sequence(const U *__restrict in, U *__restrict out) {
    v16uint32 x[4], y[2][2], w[2][2];
    AIE_LOOP_UNROLL_FULL
    for (unsigned k = 0; k < 4; ++k)
      x[k] = aie::concat(aie::load_v<V>(in + 2 * k * DIM_m),
                         aie::load_v<V>(in + (2 * k + 1) * DIM_m));
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < 2; ++r) {
      y[r][0] = ::shuffle(x[2 * r], x[2 * r + 1], T32_16x2_lo);
      y[r][1] = ::shuffle(x[2 * r], x[2 * r + 1], T32_16x2_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < 2; ++c) {
      w[c][0] = ::shuffle(y[0][c], y[1][c], T32_8x4_lo);
      w[c][1] = ::shuffle(y[0][c], y[1][c], T32_8x4_hi);
    }
    AIE_LOOP_UNROLL_FULL
    for (unsigned q = 0; q < 8; ++q)
      aie::store_v(out + q * DIM_m, aie::vector<U, 2 * V>(w[q & 1][q >> 2])
                                        .template extract<V>((q >> 1) & 1));
  }
};
#endif

// The row and column walks are fused into one counter so that the strip body
// is the innermost loop, the only one the pipeliner schedules. On AIE2 a row
// of at most two strips unrolls instead.
template <unsigned S>
static inline void transpose_blocks(const T *__restrict in, T *__restrict out) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  using Sh = Shuffles<S, BIT_WIDTH>;
  if constexpr (Sh::fits) {
    constexpr unsigned units = DIM_m / Sh::W;
    // Walked pointers rather than the index arithmetic of `in + o`.
    const T *s = in;
    T *d = out;
    unsigned c = 0;
    AIE_LOOP_NO_UNROLL
    for (unsigned i = 0; i < (DIM_n / S) * units; ++i) {
      Sh::transpose(s, d);
      bool wrap = ++c == units;
      c = wrap ? 0 : c;
      unsigned step = wrap ? (S - 1) * DIM_m + Sh::W : Sh::W;
      s += step;
      d += step;
    }
    return;
  }
#endif
  using St = Strips<S>;
  constexpr unsigned cols = DIM_m / St::W;
  if constexpr (AIE_TUNED_AIE2 && cols <= 2) {
    for (unsigned row = 0; row < DIM_n; row += S) {
      AIE_LOOP_UNROLL_FULL
      for (unsigned col = 0; col < DIM_m; col += St::W)
        St::transpose(in, out, row, col);
    }
  } else {
    unsigned row = 0, col = 0;
    for (unsigned blk = 0; blk < (DIM_n / S) * cols; ++blk) {
      St::transpose(in, out, row, col);
      col += St::W;
      if (col == DIM_m) {
        col = 0;
        row += S;
      }
    }
  }
}

extern "C" {

void transpose_4x4(T *__restrict in_ptr, T *__restrict out_ptr) {
  event0();
  transpose_blocks<4>(in_ptr, out_ptr);
  event1();
}

void transpose_8x8(T *__restrict in_ptr, T *__restrict out_ptr) {
  event0();
  transpose_blocks<8>(in_ptr, out_ptr);
  event1();
}
}
