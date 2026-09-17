//===- flash_attn_prefill.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flash_attn_prefill.h"

#include <aie_api/aie.hpp>

// Entry points for both geometries from one translation unit; see
// flash_attn_prefill.h for what separates them. A design links only the prefix
// it calls, attn_* or swa_*.
//
// The split into per-step entry points lets a core body drive the loops with an
// acquire point per object, which is what allows the Q/K/V and O legs to be
// ObjectFifos. Each takes one buffer per argument and no locks; ping/pong
// selection and synchronization belong to the design.

namespace {
constexpr int kGlobalDH = 512;
constexpr int kSlidingDH = 256;
} // namespace

extern "C" {

//===----------------------------------------------------------------------===//
// Global attention, head_dim 512
//===----------------------------------------------------------------------===//

void attn_rounds(int *L_begin_buffer, int *L_end_buffer, int *n_out) {
  rounds_impl(L_begin_buffer, L_end_buffer, n_out);
}

void attn_blocks(int *L_begin_buffer, const int i, int *n_out) {
  blocks_unbounded_impl(L_begin_buffer, i, n_out);
}

void attn_round_begin(bf16 *prev_m, bf16 *new_m, float *c, float *l, float *y) {
  round_begin_impl<kGlobalDH>(prev_m, new_m, c, l, y);
}

void attn_block_begin(bf16 *m, bf16 *prev_m) {
  block_begin_impl<kGlobalDH>(m, prev_m);
}

void attn_qk_step(bf16 *s, bf16 *__restrict q, bf16 *__restrict k, bf16 *m,
                  int *L_begin_buffer, int *window_size_buffer, const int row,
                  const int col, const int i, const int block_idx,
                  const int j) {
  qk_step_impl<kGlobalDH>(s, q, k, m, L_begin_buffer, window_size_buffer[0],
                          row, col, i, block_idx, j);
}

void attn_block_mid(bf16 *s, bf16 *m, bf16 *new_m, bf16 *prev_m, float *c,
                    float *l, float *y) {
  block_mid_impl<kGlobalDH>(s, m, new_m, prev_m, c, l, y);
}

void attn_fv_step(float *y, bf16 *s, bf16 *__restrict v, const int j) {
  fv_step_impl<kGlobalDH>(y, s, v, j);
}

void attn_block_end(bf16 *prev_m, bf16 *new_m) {
  block_end_impl<kGlobalDH>(prev_m, new_m);
}

void attn_finalize(float *l, bf16 *l_bf16) {
  finalize_impl<kGlobalDH>(l, l_bf16);
}

void attn_epilogue(bf16 *__restrict o, bf16 *l_bf16, float *y, const int c) {
  epilogue_impl<kGlobalDH>(o, l_bf16, y, c);
}

//===----------------------------------------------------------------------===//
// Sliding-window attention, head_dim 256
//===----------------------------------------------------------------------===//

void swa_rounds(int *L_begin_buffer, int *L_end_buffer, int *n_out) {
  rounds_impl(L_begin_buffer, L_end_buffer, n_out);
}

void swa_blocks(int *L_begin_buffer, int *window_size_buffer, const int i,
                int *n_out) {
  blocks_windowed_impl(L_begin_buffer, window_size_buffer[0], i, n_out);
}

void swa_round_begin(bf16 *prev_m, bf16 *new_m, float *c, float *l, float *y) {
  round_begin_impl<kSlidingDH>(prev_m, new_m, c, l, y);
}

void swa_block_begin(bf16 *m, bf16 *prev_m) {
  block_begin_impl<kSlidingDH>(m, prev_m);
}

void swa_qk_step(bf16 *s, bf16 *__restrict q, bf16 *__restrict k, bf16 *m,
                 int *L_begin_buffer, int *window_size_buffer, const int row,
                 const int col, const int i, const int block_idx, const int j) {
  qk_step_impl<kSlidingDH>(s, q, k, m, L_begin_buffer, window_size_buffer[0],
                           row, col, i, block_idx, j);
}

void swa_block_mid(bf16 *s, bf16 *m, bf16 *new_m, bf16 *prev_m, float *c,
                   float *l, float *y) {
  block_mid_impl<kSlidingDH>(s, m, new_m, prev_m, c, l, y);
}

void swa_fv_step(float *y, bf16 *s, bf16 *__restrict v, const int j) {
  fv_step_impl<kSlidingDH>(y, s, v, j);
}

void swa_block_end(bf16 *prev_m, bf16 *new_m) {
  block_end_impl<kSlidingDH>(prev_m, new_m);
}

void swa_finalize(float *l, bf16 *l_bf16) {
  finalize_impl<kSlidingDH>(l, l_bf16);
}

void swa_epilogue(bf16 *__restrict o, bf16 *l_bf16, float *y, const int c) {
  epilogue_impl<kSlidingDH>(o, l_bf16, y, c);
}

} // extern "C"
