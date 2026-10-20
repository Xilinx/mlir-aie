//===- llama_rope_int8_mh.cc --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Multi-head GQA rope: one call rotates all N_HEADS_Q Q heads (32 for
// Llama 3.2 1B at HEAD_DIM=64) and passes a per-Q-head scale tail through
// unchanged (q_scale_h, sv_inv_out_scale_h fp32 pairs, 8 B/head) so the
// downstream q-splitter + flowkv_mh kernel can find them. Body+tail
// layout is input == output. Math mirrors llama_rope_int8_dyn's
// simplified form: act_scale and 1/act_scale cancel through rope, and
// dropping them removes 1-ULP noise relative to including them.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "llama_rope_heads.h"
#include "llama_vec_fp.h"

#ifndef LLAMA_ROPE_MH_HEAD_DIM
#define LLAMA_ROPE_MH_HEAD_DIM 64
#endif
#ifndef LLAMA_ROPE_MH_N_HEADS
#define LLAMA_ROPE_MH_N_HEADS 32
#endif

static constexpr int kHD = LLAMA_ROPE_MH_HEAD_DIM;
static constexpr int kNHeads = LLAMA_ROPE_MH_N_HEADS;

static_assert(kHD % 2 == 0, "head_dim must be even");

extern "C" void llama_rope_int8_mh_dyn(int8_t *restrict x,
                                       bfloat16 *restrict cs_packed,
                                       int8_t *restrict out) {
  event0();
  llama_rope_heads<kHD, kNHeads>(x, cs_packed, out);
  event1();
}
