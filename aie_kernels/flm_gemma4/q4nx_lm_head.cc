//===- q4nx_lm_head.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Final-logits projection against a q4nx-quantized vocabulary: RMS-normalize
// the token, accumulate one weight block at a time, then apply a tanh softcap.
// One entry point per step of the caller's streaming loop. The caller owns the
// synchronization, so none of these locks or double-buffers. Geometry arrives
// as -D flags; the Q4NX_ prefix avoids R, S and T, which collide with template
// parameters in the aie2p built-in headers.
#include "../aie_kernel_utils.h"
#include "../common/zero.h"
#include "q4nx.h"
#include "rms_norm.h"
#include "utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) ||  \
    !defined(FLM_GEMMA4_LM_HEAD_DIM)
#error "q4nx_lm_head.cc needs its geometry -D defined"
#endif

// Named outside the anonymous namespace because it appears in an extern "C"
// signature.
using q4nx_block_t = q4nx_block<Q4NX_M_TILE, Q4NX_K_TILE>;

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int GROUP = Q4NX_GROUP;
constexpr int DIM = FLM_GEMMA4_LM_HEAD_DIM;

constexpr int K_BLOCKS = DIM / K_TILE;

static_assert(GROUP == 32, "a column sum covers exactly one group");
static_assert(K_TILE % GROUP == 0, "a k tile must hold whole groups");
static_assert(DIM % K_TILE == 0,
              "the RMS pass and the block loop cover whole k tiles");
static_assert(M_TILE % 16 == 0, "rows are processed 16 at a time");

} // namespace

extern "C" {

/// Once per token: normalize x in place, its RMS weight packed at x + DIM, then
/// precompute the per-32-column sums of x that every weight block reuses.
void q4nx_lm_head_rms(bfloat16 *x, bfloat16 *b_group_sums) {
  rms_norm<DIM>(x, x, x + DIM);
  for (int k = 0; k < K_BLOCKS; k++) {
    group_sums_32<K_TILE>(b_group_sums + k * (K_TILE / 32), x + k * K_TILE);
  }
}

/// Once per output tile, before the k loop.
void q4nx_lm_head_zero(float *y_acc) {
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
}

/// One weight block: y_acc += w * x[k]. k selects the slice of x and of the
/// column sums.
void q4nx_lm_head_block(const q4nx_block_t *w, const bfloat16 *x, float *y_acc,
                        const bfloat16 *b_group_sums, int k) {
  event0();
  q4nx_accumulate<M_TILE, K_TILE>(w, x + k * K_TILE, y_acc,
                                  b_group_sums + k * (K_TILE / 32));
  event1();
}

/// Once per output tile, after the k loop: y = c * tanh(y_acc / c).
void q4nx_lm_head_epilogue(bfloat16 *y, const float *y_acc,
                           const float *softcap) {
  narrow_to_bf16<M_TILE>(y, y_acc);
  aie::vector<bfloat16, M_TILE> y_vec = aie::load_v<M_TILE>(y);
  bfloat16 cap = (bfloat16)(*softcap);
  aie::accum<accfloat, M_TILE> scaled = aie::mul(y_vec, aie::inv(cap));
  y_vec = aie::tanh(scaled.template to_vector<float>());
  aie::store_v(y, aie::mul(y_vec, cap).template to_vector<bfloat16>());
}
}
