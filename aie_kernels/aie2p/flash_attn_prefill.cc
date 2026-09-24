//===- flash_attn_prefill.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flash_attn_prefill.h"

#include <aie_api/aie.hpp>

// The five entry points of one flash-attention prefill geometry; see
// flash_attn_prefill.h for the algorithm and what separates the geometries.
//
// PREFILL_HEAD_DIM picks the geometry: 512 global, 256 sliding-window. One per
// translation unit, since the factory's per-parameterization symbol prefix
// already keeps two instantiations apart in a single design.
//
// The split is per step so a core body can take an acquire point per object,
// which is what lets the Q/K/V and O legs be ObjectFifos. These carry no
// locks: synchronization belongs to the design.

#ifndef PREFILL_HEAD_DIM
#define PREFILL_HEAD_DIM 512
#endif

extern "C" {

/// Per-round init of the online-softmax accumulators.
void prefill_round_begin(bf16 *prev_m, bf16 *new_m, float *c, float *l,
                         float *y) {
  event0();
  round_begin_impl<PREFILL_HEAD_DIM>(prev_m, new_m, c, l, y);
  event1();
}

/// S = QK^T for key chunk j, masked and folded into the running row max.
/// j == 0 also seeds m from prev_m, so a block needs no opening call.
void prefill_qk_step(bf16 *s, bf16 *__restrict q, bf16 *__restrict k, bf16 *m,
                     bf16 *prev_m, int *L_begin_buffer, int *window_size_buffer,
                     const int row, const int col, const int i,
                     const int block_idx, const int j) {
  event0();
  qk_step_impl<PREFILL_HEAD_DIM>(s, q, k, m, prev_m, L_begin_buffer,
                                 window_size_buffer[0], row, col, i, block_idx,
                                 j);
  event1();
}

/// Row max, softmax, correction and the running sums for one block, then this
/// block's max becomes the next block's prev.
void prefill_block_mid(bf16 *s, bf16 *m, bf16 *new_m, bf16 *prev_m, float *c,
                       float *l, float *y) {
  event0();
  block_mid_impl<PREFILL_HEAD_DIM>(s, m, new_m, prev_m, c, l, y);
  event1();
}

/// y += S*V for key chunk j.
void prefill_fv_step(float *y, bf16 *s, bf16 *__restrict v, const int j) {
  event0();
  fv_step_impl<PREFILL_HEAD_DIM>(y, s, v, j);
  event1();
}

/// Chunk c of the output; see epilogue_impl for why l is read-write.
void prefill_epilogue(bf16 *__restrict o, float *l, float *y, const int c) {
  event0();
  epilogue_impl<PREFILL_HEAD_DIM>(o, l, y, c);
  event1();
}

} // extern "C"
