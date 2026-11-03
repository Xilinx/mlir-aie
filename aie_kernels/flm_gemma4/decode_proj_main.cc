//===- decode_proj_main.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_geometry.h"
#include "q4nx.h"
#include "utils.h"

using q4k_block_t = q4nx_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>;
constexpr int m = Q4NX_ROW_BLOCK_SIZE;
constexpr int k = Q4NX_COL_BLOCK_SIZE;

extern "C" {
void proj_zero(float *acc) { zero_256<float, m>(acc); }

void proj_accumulate(q4k_block_t *w, bf16 *x, float *acc, bf16 *sums, int row,
                     int col) {
  bf16 *group_sums = sums + col * (k / 32);
  if (row == 0)
    group_sums_32<k>(group_sums, x);
  q4nx_accumulate<m, k>(w, x, acc, group_sums);
}

void proj_store(bf16 *y, float *acc) { narrow_to_bf16<m>(y, acc); }

// Only the sending core produces this object. Its neighbour hands over its
// row block through a shared-memory ObjectFifo.
void proj_pack(bf16 *y, float *acc, bf16 *other, int packet_id) {
  *reinterpret_cast<uint32 *>(y + 14) = packet_id;
  narrow_to_bf16<m>(y + 16, acc);
  copy_vectorized<bf16, m>(y + 16 + m, other);
}
}
