//===- decode_glu.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_layout.h"
#include "lut_based_ops.h"
#include "utils.h"

template <int L>
void pseduo_glu(bf16 *y, const bf16 *x) {
  bf16 *gate_ptr = const_cast<bf16 *>(x) + (L / 2);
  bf16 *hid_ptr = const_cast<bf16 *>(x);
  bf16 *y_ptr = y;
  for (int i = 0; i < L / 2; i += 16) {
    aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
    aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
    gate_vec = getGeluBf16(gate_vec);

    aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
    aie::store_v(y_ptr + i, y_vec);
  }
}

constexpr int x_prod_lock = FLM_GEMMA4_DECODE_GLU_X_PROD_LOCK;
constexpr int x_cons_lock = FLM_GEMMA4_DECODE_GLU_X_CONS_LOCK;
constexpr int y_prod_lock = FLM_GEMMA4_DECODE_GLU_Y_PROD_LOCK;
constexpr int y_cons_lock = FLM_GEMMA4_DECODE_GLU_Y_CONS_LOCK;
extern "C" {
static bool is_x_ping = false;
static bool is_y_ping = false;
void glu(bf16 *y, const bf16 *x_ping, const bf16 *x_pong, bf16 *y_ping,
         bf16 *y_pong, int *SKIP_KV) {
  static_assert((2 * INTERMEDIATE_SIZE) % GLU_SLICE == 0,
                "GLU_SLICE must divide 2 * INTERMEDIATE_SIZE");
  constexpr int chunks =
      2 * INTERMEDIATE_SIZE /
      GLU_SLICE; // number of chunks in one GLU, each chunk is GLU_SLICE in size

  if (SKIP_KV[0] == 0) {
    bf16 *y_it = y;
    for (int i = 0; i < chunks; i++) {
      is_x_ping = !is_x_ping;
      const bf16 *x_using = is_x_ping ? x_ping : x_pong;
      _lock_acquire_p(x_using, x_cons_lock);
      pseduo_glu<GLU_SLICE>(y_it, x_using);
      y_it += GLU_SLICE / 2;
      _lock_release_p((void *)x_using, x_prod_lock);
    }

    for (int i = 0; i < O_DOWN_REPEATS; i++) {
      for (int j = 0; j < INTERMEDIATE_SIZE / GLU_SLICE * 2; j++) {
        is_y_ping = !is_y_ping;
        bf16 *y_using = is_y_ping ? y_ping : y_pong;
        _lock_acquire_p(y_using, y_prod_lock);
        copy_vectorized<bf16, GLU_SLICE / 2>(y_using, y + j * GLU_SLICE / 2);
        _lock_release_p(y_using, y_cons_lock);
      }
    }
  } else {
    bf16 *y_it = y;
    for (int i = 0; i < chunks * 2; i++) {
      is_x_ping = !is_x_ping;
      const bf16 *x_using = is_x_ping ? x_ping : x_pong;
      _lock_acquire_p(x_using, x_cons_lock);
      pseduo_glu<GLU_SLICE>(y_it, x_using);
      y_it += GLU_SLICE / 2;
      _lock_release_p((void *)x_using, x_prod_lock);
    }

    for (int i = 0; i < O_DOWN_REPEATS; i++) {
      for (int j = 0; j < INTERMEDIATE_SIZE / GLU_SLICE * 4; j++) {
        is_y_ping = !is_y_ping;
        bf16 *y_using = is_y_ping ? y_ping : y_pong;
        _lock_acquire_p(y_using, y_prod_lock);
        copy_vectorized<bf16, GLU_SLICE / 2>(y_using, y + j * GLU_SLICE / 2);
        _lock_release_p(y_using, y_cons_lock);
      }
    }
  }
  bf16 *y_it = y;
}
}