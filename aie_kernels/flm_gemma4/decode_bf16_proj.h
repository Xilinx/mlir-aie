//===- decode_bf16_proj.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_FLM_GEMMA4_DECODE_BF16_PROJ_H
#define AIE_KERNELS_FLM_GEMMA4_DECODE_BF16_PROJ_H
#include "decode_geometry.h"
#include "utils.h"

// These kernels need a Peano that contains llvm-aie PR #1275
// (https://github.com/Xilinx/llvm-aie/pull/1275, merged 2026-09-01; every
// nightly since contains it). Without it, AIE2P codegen does not write the
// float accumulator back on the loop back edge once _mvm_bf16_bf16 is inlined
// into the frame that declares y_acc. Only the last partial sum survives.
template <int M, int K>
void _mvm_bf16_bf16(bf16 *w, bf16 *x, float *y_acc) {
  aie::vector<float, M> y_acc_vec = aie::load_v<M>(y_acc);
  aie::accum<accfloat, M> acc;

  constexpr int x_vec_size = 32;
  bf16 *w_ptr = w;
  acc.from_vector(y_acc_vec);
  for (int i = 0; i < K / x_vec_size; i++) {
    aie::vector<bf16, x_vec_size> b_col =
        aie::load_v<x_vec_size>(x + i * x_vec_size);

    for (int j = 0; j < x_vec_size; j++) {
      aie::vector<bf16, M> w_col = aie::load_v<M>(w_ptr);
      bf16 b_col_j = b_col[j];
      acc = aie::mac(acc, w_col, b_col_j);
      w_ptr += M;
    }
  }
  aie::store_v(y_acc, acc.template to_vector<float>());
}

template <int M, int K>
void linear_proj(bf16 *y, bf16 *w_ping, bf16 *w_pong, bf16 *x, float *y_acc,
                 PingPong &w_pingpong, const int w_prod_lock_id,
                 const int w_cons_lock_id) {

  constexpr int m = BF16_PROJ_M_BLOCK;
  constexpr int k = BF16_PROJ_K_BLOCK;

  for (int i = 0; i < M / m; i++) {
    zero_256<float, m>(y_acc);
    for (int j = 0; j < K / k; j++) {
      bf16 *w_using = w_pingpong.next(w_ping, w_pong);
      bf16 *x_using = x + j * k;
      _lock_acquire(w_cons_lock_id);
      _mvm_bf16_bf16<m, k>(w_using, x_using, y_acc);
      _lock_release(w_prod_lock_id);
    }

    bf16 *y_using = y + i * m;
    aie::vector<float, m> y_acc_vec = aie::load_v<m>(y_acc);
    aie::accum<accfloat, m> c_acc;
    c_acc.from_vector(y_acc_vec);
    aie::vector<bf16, m> y_vec = c_acc.template to_vector<bf16>();
    aie::store_v(y_using, y_vec);
  }
}
#endif // AIE_KERNELS_FLM_GEMMA4_DECODE_BF16_PROJ_H