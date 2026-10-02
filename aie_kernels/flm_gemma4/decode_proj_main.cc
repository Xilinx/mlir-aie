//===- decode_proj_main.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_geometry.h"
#include "decode_layout.h"
#include "q4nx.h"
#include "utils.h"

using q4k_block_t = q4nx_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>;

constexpr int pkt_id_to_rope = 1;
constexpr int pkt_id_to_swa_rope = 2;
constexpr int pkt_id_to_rms = 4;
constexpr int pkt_id_to_glu = 8;

constexpr int x_prod_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_X_PROD_LOCK;
constexpr int x_cons_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_X_CONS_LOCK;
constexpr int w_prod_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_W_PROD_LOCK;
constexpr int w_cons_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_W_CONS_LOCK;
constexpr int y_prod_ping_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_Y_PROD_PING_LOCK;
constexpr int y_prod_pong_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_Y_PROD_PONG_LOCK;
constexpr int rtp_available_lock =
    FLM_GEMMA4_DECODE_PROJ_MAIN_RTP_AVAILABLE_LOCK;
constexpr int y_cons_ping_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_Y_CONS_PING_LOCK;
constexpr int y_cons_pong_lock = FLM_GEMMA4_DECODE_PROJ_MAIN_Y_CONS_PONG_LOCK;
// proj_main runs the projections of one layer in order: qkv (to RoPE), o (to
// rms_residual), up/gate (to glu) and down (to rms_residual).
PingPong y_pingpong;
PingPong w_pingpong;
PingPong x_pingpong;

void linear_proj_iD(int M, int K, bf16 *x_ping, bf16 *x_pong,
                    q4k_block_t *w_ping, q4k_block_t *w_pong, bf16 *y_ping,
                    bf16 *y_pong, float *y_acc, bf16 *b_col_reduce_add,
                    const int32_t send_x_output, const uint32 pkt_id) {

  constexpr int m = Q4NX_ROW_BLOCK_SIZE;
  constexpr int k = Q4NX_COL_BLOCK_SIZE;

  uint32 *pkt_id_ping = reinterpret_cast<uint32 *>(y_ping + 14);
  uint32 *pkt_id_pong = reinterpret_cast<uint32 *>(y_pong + 14);

  for (int i = 0; i < M / m; i++) {
    zero_256<float, m>(y_acc);
    for (int j = 0; j < K / k; j++) {
      q4k_block_t *w_using = w_pingpong.next(w_ping, w_pong);
      bf16 *x_using = x_pingpong.next(x_ping, x_pong);
      _lock_acquire_p(w_using, w_cons_lock);
      _lock_acquire_p(x_using, x_cons_lock);
      bf16 *b_col_reduce_add_ptr = b_col_reduce_add + j * (k / 32);
      // The group sums of x depend only on j. Compute them for the first row
      // block and reuse them for the others.
      if (i == 0) {
        group_sums_32<k>(b_col_reduce_add_ptr, x_using);
      }
      q4nx_accumulate<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
      _lock_release_p(w_using, w_prod_lock);
      _lock_release_p(x_using, x_prod_lock);
    }
    bf16 *y_using = y_pingpong.next(y_ping, y_pong);

    if (send_x_output != 0) {
      uint32 *pkt_id_using = y_pingpong.is_ping ? pkt_id_ping : pkt_id_pong;
      *pkt_id_using = pkt_id;
      if (y_pingpong.is_ping) {
        _lock_acquire_p(y_using, y_prod_ping_lock);
      } else {
        _lock_acquire_p(y_using, y_prod_pong_lock);
      }
      narrow_to_bf16<m>(y_using + 16, y_acc);
      if (y_pingpong.is_ping) {
        _lock_release_p(y_using, y_cons_ping_lock);
      } else {
        _lock_release_p(y_using, y_cons_pong_lock);
      }
    } else {
      if (y_pingpong.is_ping) {
        _down_lock_acquire_p(y_using, y_prod_ping_lock);
      } else {
        _down_lock_acquire_p(y_using, y_prod_pong_lock);
      }
      narrow_to_bf16<m>(y_using + 16 + m,
                        y_acc); // offset of 16+m, 16 for packed_it
      if (y_pingpong.is_ping) {
        _down_lock_release_p(y_using, y_cons_ping_lock);
      } else {
        _down_lock_release_p(y_using, y_cons_pong_lock);
      }
    }
  }
}

extern "C" {

void proj_main(bf16 *y_ping, q4k_block_t *w_ping, bf16 *x_ping, bf16 *y_pong,
               q4k_block_t *w_pong, bf16 *x_pong, int *IS_SWA, int *SKIP_KV,
               int send_x_output) {
  alignas(aie::vector_decl_align) float y_acc[Q4NX_ROW_BLOCK_SIZE];
#ifdef DOUBLE_WIDE_MLP
  alignas(aie::vector_decl_align)
      bfloat16 b_col_reduce_add[2 * INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE];
#else
  alignas(aie::vector_decl_align)
      bfloat16 b_col_reduce_add[INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE];
#endif

  _lock_acquire_p(IS_SWA, rtp_available_lock);
  if (IS_SWA[0]) {
    if (SKIP_KV[0] == 0) {
      static_assert(
          (SWA_DQ + SWA_DK + SWA_DV) % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0,
          "M must be divisible by MVM_CORES");
      linear_proj_iD((SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES, MODEL_DIM, x_ping,
                     x_pong, w_ping, w_pong, y_ping, y_pong, y_acc,
                     b_col_reduce_add, send_x_output, pkt_id_to_swa_rope);
    } else {
      // With SKIP_KV, compute only the q projection.
      static_assert(SWA_DQ % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0,
                    "M must be divisible by MVM_CORES");
      linear_proj_iD(SWA_DQ / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping,
                     w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                     send_x_output, pkt_id_to_swa_rope);
    }

    linear_proj_iD(MODEL_DIM / MVM_CORES, SWA_DQ, x_ping, x_pong, w_ping,
                   w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                   send_x_output, pkt_id_to_rms);
  } else {
    if (SKIP_KV[0] == 0) {
      static_assert((DQ + DK + DV) % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0,
                    "M must be divisible by MVM_CORES");
      linear_proj_iD((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong,
                     w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                     send_x_output, pkt_id_to_rope);
    } else {
      // With SKIP_KV, compute only the q projection.
      static_assert(DQ % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0,
                    "M must be divisible by MVM_CORES");
      linear_proj_iD(DQ / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong,
                     y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output,
                     pkt_id_to_rope);
    }

    linear_proj_iD(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong,
                   y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output,
                   pkt_id_to_rms);
  }
#ifdef DOUBLE_WIDE_MLP
  if (SKIP_KV[0]) {
    linear_proj_iD(4 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong,
                   w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                   send_x_output, pkt_id_to_glu);

    linear_proj_iD(MODEL_DIM / MVM_CORES, 2 * INTERMEDIATE_SIZE, x_ping, x_pong,
                   w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                   send_x_output, pkt_id_to_rms);
  } else {
    linear_proj_iD(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong,
                   w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                   send_x_output, pkt_id_to_glu);

    linear_proj_iD(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong,
                   w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                   send_x_output, pkt_id_to_rms);
  }
#else
  linear_proj_iD(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong,
                 w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                 send_x_output, pkt_id_to_glu);

  linear_proj_iD(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong,
                 w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add,
                 send_x_output, pkt_id_to_rms);
#endif
}
}