//===- decode_glu.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_layout.h"
#include "lut_based_ops.h"
#include "utils.h"

// getGeluBf16's linear_approx written out over 32 lanes: 64 segments of 1/8
// over [-4, 4), each offset + slope * x with x itself unclamped, so the end
// segments extend the curve. linear_approx offsets the table pointers by the
// centring bias; it goes on the index instead, the same address, because an
// offset table pointer loses the reads' memory operands.
static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
gelu_lut(aie::vector<bfloat16, 32> x) {
  constexpr int bias_bytes = 32 << 4;
  aie::accum<accfloat, 32> offset;
  aie::vector<bfloat16, 32> slope;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    aie::vector<int32, 16> index = ::bfloat16_to_int(x.extract<16>(h), 7);
    index = aie::min(aie::max(index, -bias_bytes), bias_bytes - 1);
    index = aie::add(index, bias_bytes);
    v32bfloat16 coeff0, coeff1;
    ::load_lut_2x_float(gelu_lut_ab, gelu_lut_cd, index, coeff0, coeff1);
    offset.insert(h, aie::accum<accfloat, 16>(
                         (v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
    slope.insert(
        h, aie::vector<bfloat16, 32>(::shuffle(coeff0, coeff1, T16_16x4_lo))
               .extract<16>(1));
  }
  return aie::accum<accfloat, 32>(::mac_elem_32(slope, x, offset))
      .to_vector<bfloat16>();
}

// y = gelu(gate) * up. The table reads run in a loop of their own, which
// pipelines; gelu's bf16 result goes through y and comes back unchanged.
template <int L>
void pseduo_glu(bf16 *__restrict y, const bf16 *__restrict x) {
  static_assert((L / 2) % 64 == 0);
  const bf16 *__restrict gate = x + L / 2;
  for (int i = 0; i < L / 2; i += 32)
    aie::store_v(y + i, gelu_lut(aie::load_v<32>(gate + i)));
  // In place over y, two vectors a trip with both loaded before either store,
  // so one trip's stores hold back the next trip's loads once per 64 lanes.
  for (int i = 0; i < L / 2; i += 64) {
    const aie::vector<bfloat16, 32> g0 = aie::load_v<32>(y + i);
    const aie::vector<bfloat16, 32> g1 = aie::load_v<32>(y + i + 32);
    const aie::vector<bfloat16, 32> u0 = aie::load_v<32>(x + i);
    const aie::vector<bfloat16, 32> u1 = aie::load_v<32>(x + i + 32);
    aie::store_v(y + i, aie::mul(g0, u0).to_vector<bfloat16>());
    aie::store_v(y + i + 32, aie::mul(g1, u1).to_vector<bfloat16>());
  }
}

constexpr int x_prod_lock = FLM_GEMMA4_DECODE_GLU_X_PROD_LOCK;
constexpr int x_cons_lock = FLM_GEMMA4_DECODE_GLU_X_CONS_LOCK;
constexpr int y_prod_lock = FLM_GEMMA4_DECODE_GLU_Y_PROD_LOCK;
constexpr int y_cons_lock = FLM_GEMMA4_DECODE_GLU_Y_CONS_LOCK;
constexpr int rtp_lock = FLM_GEMMA4_DECODE_GLU_RTP_LOCK;
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

  _lock_acquire(rtp_lock);

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