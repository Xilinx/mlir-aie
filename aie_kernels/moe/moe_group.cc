//===- moe_group.cc ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Groups a chunk's routed rows by expert, for a mixture of experts' sparse
// prompt: each expert's assignments take a run of whole GROUP_TILE-slot
// blocks, in expert order, so a GEMM over the grouped rows changes expert
// only between row blocks.
//
// Two passes over the chunk's router ids, GROUP_ROWS rows of GROUP_TOP_K a
// call: moe_group_count counts each expert's assignments, moe_group_blocks
// lays out the blocks, and moe_group_where gives each assignment its slot.
// The state is GROUP_EXPERTS counts, then GROUP_EXPERTS cursors.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GROUP_ROWS
#error "GROUP_ROWS (ids rows a call takes) must be defined"
#endif
#ifndef GROUP_TOP_K
#error "GROUP_TOP_K (experts a row is routed to) must be defined"
#endif
#ifndef GROUP_EXPERTS
#error "GROUP_EXPERTS must be defined"
#endif
#ifndef GROUP_TILE
#error "GROUP_TILE (slots a block holds) must be defined"
#endif
#ifndef GROUP_BLOCKS
#error "GROUP_BLOCKS (blocks the layout holds) must be defined"
#endif

namespace {

// An id clipped into the experts, a negative one counting from the end.
int32_t group_expert(int32_t id) {
  if (id < -GROUP_EXPERTS)
    id = -GROUP_EXPERTS;
  if (id >= GROUP_EXPERTS)
    id = GROUP_EXPERTS - 1;
  return id < 0 ? id + GROUP_EXPERTS : id;
}

// The rows of this call's GROUP_ROWS, from row `first`, that `valid` holds.
int32_t group_rows(int32_t first, int32_t valid) {
  int32_t rows = valid - first;
  if (rows < 0)
    return 0;
  return rows > GROUP_ROWS ? GROUP_ROWS : rows;
}

} // namespace

extern "C" {

// ids: GROUP_ROWS x GROUP_TOP_K expert ids, chunk rows `first` on; state:
// 2 * GROUP_EXPERTS int32, cleared by the call whose `first` is 0. Rows from
// `valid` on are not counted.
void moe_group_count(const int32_t *ids, int32_t *state, int32_t first,
                     int32_t valid) {
  event0();
  if (first == 0)
    for (int32_t e = 0; e < GROUP_EXPERTS; ++e)
      state[e] = 0;
  const int32_t rows = group_rows(first, valid);
  for (int32_t a = 0; a < rows * GROUP_TOP_K; ++a)
    ++state[group_expert(ids[a])];
  event1();
}

// blocks: GROUP_BLOCKS int32, each block's expert, every expert's blocks in
// expert order and those past the last one the last expert's (0 when none
// was routed to). Each expert's cursor becomes its first slot.
void moe_group_blocks(int32_t *state, int32_t *blocks) {
  event0();
  int32_t b = 0;
  int32_t last = 0;
  for (int32_t e = 0; e < GROUP_EXPERTS; ++e) {
    state[GROUP_EXPERTS + e] = b * GROUP_TILE;
    const int32_t n = (state[e] + GROUP_TILE - 1) / GROUP_TILE;
    for (int32_t i = 0; i < n; ++i)
      blocks[b++] = e;
    if (n)
      last = e;
  }
  for (; b < GROUP_BLOCKS; ++b)
    blocks[b] = last;
  event1();
}

// where: GROUP_ROWS x GROUP_TOP_K, each assignment's slot, an expert's
// assignments in row order; 0 for rows from `valid` on.
void moe_group_where(const int32_t *ids, int32_t *state, int32_t *where,
                     int32_t first, int32_t valid) {
  event0();
  const int32_t assigned = group_rows(first, valid) * GROUP_TOP_K;
  for (int32_t a = 0; a < GROUP_ROWS * GROUP_TOP_K; ++a)
    where[a] = a < assigned ? state[GROUP_EXPERTS + group_expert(ids[a])]++ : 0;
  event1();
}

} // extern "C"
