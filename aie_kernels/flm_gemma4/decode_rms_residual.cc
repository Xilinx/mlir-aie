//===- decode_rms_residual.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_geometry.h"
#include "decode_layout.h"
#include "rms_norm.h"
#include "utils.h"

extern "C" {

void rms_residual(bf16 *restrict y, bf16 *restrict x_ping,
                  bf16 *restrict x_pong, bf16 *restrict y_final,
                  bf16 *restrict w, bf16 *restrict x_temp_buf, int *IS_SWA,
                  int *SKIP_KV) {
  constexpr int w_prod_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_W_PROD_LOCK;
  constexpr int w_cons_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_W_CONS_LOCK;
  constexpr int y_prod_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_Y_PROD_LOCK;
  constexpr int y_cons_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_Y_CONS_LOCK;
  constexpr int x_prod_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_X_PROD_LOCK;
  constexpr int x_cons_lock = FLM_GEMMA4_DECODE_RMS_RESIDUAL_X_CONS_LOCK;
  constexpr int rtp_available_lock =
      FLM_GEMMA4_DECODE_RMS_RESIDUAL_RTP_AVAILABLE_LOCK;
  constexpr int lm_head_out_prod_lock =
      FLM_GEMMA4_DECODE_RMS_RESIDUAL_LM_HEAD_OUT_PROD_LOCK;
  constexpr int lm_head_out_cons_lock =
      FLM_GEMMA4_DECODE_RMS_RESIDUAL_LM_HEAD_OUT_CONS_LOCK;
  uint32_t *pkt_id_ptr = reinterpret_cast<uint32_t *>(y + 14);
  bf16 *x;
  bf16 *w_input_layernorm = w;
  bf16 *w_post_attn_layernorm = w + MODEL_DIM;
  bf16 *w_pre_feedforward_layernorm = w + MODEL_DIM * 2;
  bf16 *w_post_feedforward_layer_norm = w + MODEL_DIM * 3;
  bf16 *x_buf = x_temp_buf;
  bf16 *temp_buf = x_temp_buf + MODEL_DIM;

  static PingPong x_pingpong;

  _lock_acquire_p(IS_SWA, rtp_available_lock);
  // core-local, so it may precede the lock acquire
  x = x_pingpong.next(x_ping, x_pong);
  _lock_acquire_p(x, x_cons_lock);
  _lock_acquire_p(w, w_cons_lock);
  if (IS_SWA[0] == 0) {
    // input layernorm
    *pkt_id_ptr = 0;
    copy_vectorized<bf16, MODEL_DIM>(x_buf, x);
    rms_norm<MODEL_DIM>(y + 16, x_buf, w_input_layernorm);
    _lock_release_p(x, x_prod_lock);
    if (SKIP_KV[0] == 0) {
      _lock_release_p(y, y_cons_lock, QKV_REPEATS);
    } else {
      _lock_release_p(y, y_cons_lock, Q_REPEATS);
    }

    _lock_acquire_p(x, x_cons_lock);

    if (SKIP_KV[0] == 0) {
      _lock_acquire_p(y, y_prod_lock, QKV_REPEATS);
    } else {
      _lock_acquire_p(y, y_prod_lock, Q_REPEATS);
    }
  } else {
    // input layernorm
    *pkt_id_ptr = 0;

    copy_vectorized<bf16, MODEL_DIM>(x_buf, x);
    rms_norm<MODEL_DIM>(y + 16, x_buf, w_input_layernorm);
    _lock_release_p(x, x_prod_lock);
    if (SKIP_KV[0] == 0) {
      _lock_release_p(y, y_cons_lock, SWA_QKV_REPEATS);
    } else {
      _lock_release_p(y, y_cons_lock, SWA_Q_REPEATS);
    }

    _lock_acquire_p(x, x_cons_lock);

    if (SKIP_KV[0] == 0) {
      _lock_acquire_p(y, y_prod_lock, SWA_QKV_REPEATS);
    } else {
      _lock_acquire_p(y, y_prod_lock, SWA_Q_REPEATS);
    }
  }

  *pkt_id_ptr = 0;
  x = x_pingpong.next(x_ping, x_pong);
  rms_norm<MODEL_DIM>(y + 16, x, w_post_attn_layernorm);
  residual_add<MODEL_DIM>(temp_buf, x_buf, y + 16);
  // pre-feedforward layernorm
  copy_vectorized<bf16, MODEL_DIM>(x_buf, temp_buf);
  rms_norm<MODEL_DIM>(y + 16, temp_buf, w_pre_feedforward_layernorm);
  _lock_release_p(x, x_prod_lock);
#ifdef DOUBLE_WIDE_MLP
  if (SKIP_KV[0] == 0) {
    _lock_release_p(y, y_cons_lock, UP_GATE_REPEATS);
  } else {
    _lock_release_p(y, y_cons_lock, UP_GATE_REPEATS * 2);
  }
#else
  _lock_release_p(y, y_cons_lock, UP_GATE_REPEATS);
#endif

  _lock_acquire_p(x, x_cons_lock);

#ifdef DOUBLE_WIDE_MLP
  if (SKIP_KV[0] == 0) {
    _lock_acquire_p(y, y_prod_lock, UP_GATE_REPEATS);
  } else {
    _lock_acquire_p(y, y_prod_lock, UP_GATE_REPEATS * 2);
  }
#else
  _lock_acquire_p(y, y_prod_lock, UP_GATE_REPEATS);
#endif
  x = x_pingpong.next(x_ping, x_pong);
  _lock_acquire_p(y_final, lm_head_out_prod_lock, 1);
  rms_norm<MODEL_DIM>(temp_buf, x, w_post_feedforward_layer_norm);
  residual_add<MODEL_DIM>(y_final, x_buf, temp_buf);
  _lock_release_p(w, w_prod_lock);
  _lock_release_p(x, x_prod_lock);
  _lock_release_p(y_final, lm_head_out_cons_lock, 1);
}
}