//===- gemm_atb_layout.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Host-side data-layout helpers for the asymmetric-tile-buffering GEMM
// designs. The B operand on the device is read as a 1D stream of v8bfp16ebs8
// vectors organized by (L1_K-block, L1_N-block) column-major, with each
// L1 tile pre-shuffled into 1x2 super-blocks of 8x8 column-major sub-blocks
// so a single VMAC issue can stream contiguously.
//
//===----------------------------------------------------------------------===//

#ifndef GEMM_ATB_LAYOUT_H
#define GEMM_ATB_LAYOUT_H

#include <cassert>
#include <vector>

namespace gemm_atb {

// Shuffle a (rows x cols) row-major float matrix into 1x2 row-major
// super-blocks of 8x8 column-major sub-blocks. The output is the same number
// of floats, arranged so the VMAC unit's 8x8 BFP16 inputs are contiguous.
inline std::vector<float>
layout_transpose_1x2_8x8block(const std::vector<float> &input, int rows,
                              int cols) {
  assert(rows % 8 == 0 && "rows must be divisible by 8");
  assert(cols % 8 == 0 && "cols must be divisible by 8");
  assert((cols / 8) % 2 == 0 && "cols/8 must be divisible by 2 for 1x2 layout");
  std::vector<float> output(rows * cols);
  int block_rows = rows / 8;
  int block_cols = cols / 8;
  int output_idx = 0;
  // Iterate 1x2 super-blocks (two horizontally-stacked 8x8 blocks).
  for (int super_block_col = 0; super_block_col < block_cols;
       super_block_col += 2) {
    for (int super_block_row = 0; super_block_row < block_rows;
         super_block_row++) {
      for (int block_in_super = 0;
           block_in_super < std::min(2, block_cols - super_block_col);
           block_in_super++) {
        int current_block_row = super_block_row;
        int current_block_col = super_block_col + block_in_super;
        // Within each 8x8 block: column-major.
        for (int col_in_block = 0; col_in_block < 8; col_in_block++) {
          for (int row_in_block = 0; row_in_block < 8; row_in_block++) {
            int orig_row = current_block_row * 8 + row_in_block;
            int orig_col = current_block_col * 8 + col_in_block;
            int orig_idx = orig_row * cols + orig_col;
            output[output_idx++] = input[orig_idx];
          }
        }
      }
    }
  }
  return output;
}

// Same shuffle but applied tile-by-tile across an outer (rows x cols) matrix
// of L1_block_k x L1_block_n tiles, with the tiles emitted in column-major
// order (outer column-major, inner per-tile 1x2_8x8block).
inline std::vector<float>
layout_transpose_L1_1x2_8x8block(const std::vector<float> &input, int rows,
                                 int cols, int L1_block_k, int L1_block_n) {
  assert(rows % L1_block_k == 0 && "rows must be divisible by L1_block_k");
  assert(cols % L1_block_n == 0 && "cols must be divisible by L1_block_n");
  assert(L1_block_k % 8 == 0 && "L1_block_k must be divisible by 8");
  assert(L1_block_n % 8 == 0 && "L1_block_n must be divisible by 8");
  assert((L1_block_n / 8) % 2 == 0 &&
         "L1_block_n/8 must be divisible by 2 for the 1x2 layout");

  std::vector<float> output(rows * cols);
  int L1_rows = rows / L1_block_k;
  int L1_cols = cols / L1_block_n;
  int output_idx = 0;

  // Outer order: column-major over L1 tiles.
  for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
    for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
      // Extract the current L1 tile.
      std::vector<float> tile(L1_block_k * L1_block_n);
      for (int i = 0; i < L1_block_k; i++) {
        for (int j = 0; j < L1_block_n; j++) {
          int orig_row = L1_row * L1_block_k + i;
          int orig_col = L1_col * L1_block_n + j;
          tile[i * L1_block_n + j] = input[orig_row * cols + orig_col];
        }
      }
      // Inner shuffle for this tile.
      std::vector<float> shuffled =
          layout_transpose_1x2_8x8block(tile, L1_block_k, L1_block_n);
      for (size_t k = 0; k < shuffled.size(); k++) {
        output[output_idx++] = shuffled[k];
      }
    }
  }
  return output;
}

// --- A input shuffle (used by the pure-bfp16 configs) ----------------------
// A is laid out as L1 tiles in row-major order across (M, K). Within each
// L1 tile, the inner pattern is 2x1 vertically-stacked super-blocks of 8x8
// row-major sub-blocks. This is different from B's shuffle (which is 1x2
// horizontally-stacked super-blocks of 8x8 column-major sub-blocks); the two
// patterns reflect how the MAC unit's two input vectors index into their
// respective register banks.

inline std::vector<float> layout_A_2x1_8x8block(const std::vector<float> &input,
                                                int rows, int cols) {
  assert(rows % 8 == 0 && "rows must be divisible by 8");
  assert(cols % 8 == 0 && "cols must be divisible by 8");
  std::vector<float> output(rows * cols);
  int block_rows = rows / 8;
  int block_cols = cols / 8;
  assert(block_rows % 2 == 0 &&
         "block_rows must be divisible by 2 for 2x1 layout");
  int output_idx = 0;
  for (int super_block_row = 0; super_block_row < block_rows;
       super_block_row += 2) {
    for (int super_block_col = 0; super_block_col < block_cols;
         super_block_col++) {
      // 2x1 super-block: two 8x8 sub-blocks stacked vertically.
      for (int block_in_super = 0; block_in_super < 2; block_in_super++) {
        int cbr = super_block_row + block_in_super;
        int cbc = super_block_col;
        // 8x8 sub-block: row-major.
        for (int rib = 0; rib < 8; rib++) {
          for (int cib = 0; cib < 8; cib++) {
            int orig_row = cbr * 8 + rib;
            int orig_col = cbc * 8 + cib;
            output[output_idx++] = input[orig_row * cols + orig_col];
          }
        }
      }
    }
  }
  return output;
}

inline std::vector<float>
layout_A_L1_2x1_8x8block(const std::vector<float> &input, int rows, int cols,
                         int L1_block_m, int L1_block_k) {
  assert(rows % L1_block_m == 0 && "rows must be divisible by L1_block_m");
  assert(cols % L1_block_k == 0 && "cols must be divisible by L1_block_k");
  assert(L1_block_m % 8 == 0 && "L1_block_m must be divisible by 8");
  assert(L1_block_k % 8 == 0 && "L1_block_k must be divisible by 8");
  assert((L1_block_m / 8) % 2 == 0 &&
         "L1_block_m/8 must be divisible by 2 for the 2x1 layout");
  std::vector<float> output(rows * cols);
  int L1_rows = rows / L1_block_m;
  int L1_cols = cols / L1_block_k;
  int output_idx = 0;
  // Outer L1 traversal: row-major.
  for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
    for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
      std::vector<float> tile(L1_block_m * L1_block_k);
      for (int i = 0; i < L1_block_m; i++) {
        for (int j = 0; j < L1_block_k; j++) {
          int orig_row = L1_row * L1_block_m + i;
          int orig_col = L1_col * L1_block_k + j;
          tile[i * L1_block_k + j] = input[orig_row * cols + orig_col];
        }
      }
      std::vector<float> shuffled =
          layout_A_2x1_8x8block(tile, L1_block_m, L1_block_k);
      for (size_t i = 0; i < shuffled.size(); i++)
        output[output_idx++] = shuffled[i];
    }
  }
  return output;
}

// --- C-output unshuffle (inverse of the input pattern, for verification) ----
// The pure-bfp16 ATB designs (configs 2 and 3) emit C in a hierarchical
// layout: L1 blocks of `(L1_block_m, L1_block_n)` floats arranged row-major
// across the (M, N) result, and within each L1 block, 2x2 super-blocks of
// 8x8 row-major sub-blocks. The two helpers below convert that back to plain
// row-major so the host can verify against a CPU reference.

inline std::vector<float>
layout_inverse_C_2x2_8x8block(const std::vector<float> &input, int L1_block_m,
                              int L1_block_n) {
  std::vector<float> output(L1_block_m * L1_block_n);
  int input_idx = 0;
  int blocks_per_row = L1_block_n / 8;
  int blocks_per_col = L1_block_m / 8;
  for (int super_block_row = 0; super_block_row < blocks_per_col;
       super_block_row += 2) {
    for (int super_block_col = 0; super_block_col < blocks_per_row;
         super_block_col += 2) {
      // Order within each 2x2 super-block: [0,0], [0,1], [1,0], [1,1].
      for (int block_row = 0; block_row < 2; block_row++) {
        for (int block_col = 0; block_col < 2; block_col++) {
          int cbr = super_block_row + block_row;
          int cbc = super_block_col + block_col;
          for (int rib = 0; rib < 8; rib++) {
            for (int cib = 0; cib < 8; cib++) {
              int out_row = cbr * 8 + rib;
              int out_col = cbc * 8 + cib;
              output[out_row * L1_block_n + out_col] = input[input_idx++];
            }
          }
        }
      }
    }
  }
  return output;
}

inline std::vector<float>
layout_inverse_C_L1_2x2_8x8block(const std::vector<float> &input, int M, int N,
                                 int L1_block_m, int L1_block_n) {
  assert(M % L1_block_m == 0 && "M must be divisible by L1_block_m");
  assert(N % L1_block_n == 0 && "N must be divisible by L1_block_n");
  assert(L1_block_m % 16 == 0 &&
         "L1_block_m must be divisible by 16 for 2x2 8x8 blocks");
  assert(L1_block_n % 16 == 0 &&
         "L1_block_n must be divisible by 16 for 2x2 8x8 blocks");

  std::vector<float> output(M * N);
  int L1_rows = M / L1_block_m;
  int L1_cols = N / L1_block_n;
  int input_idx = 0;
  // L1-tile order: row-major (matches the output dispatch sequence in
  // the IRON design's runtime_sequence).
  for (int L1_row = 0; L1_row < L1_rows; L1_row++) {
    for (int L1_col = 0; L1_col < L1_cols; L1_col++) {
      int L1_block_size = L1_block_m * L1_block_n;
      std::vector<float> tile(L1_block_size);
      for (int i = 0; i < L1_block_size; i++)
        tile[i] = input[input_idx++];
      std::vector<float> unshuffled =
          layout_inverse_C_2x2_8x8block(tile, L1_block_m, L1_block_n);
      for (int i = 0; i < L1_block_m; i++) {
        for (int j = 0; j < L1_block_n; j++) {
          int out_row = L1_row * L1_block_m + i;
          int out_col = L1_col * L1_block_n + j;
          output[out_row * N + out_col] = unshuffled[i * L1_block_n + j];
        }
      }
    }
  }
  return output;
}

} // namespace gemm_atb

#endif // GEMM_ATB_LAYOUT_H
