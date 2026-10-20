//===- selective_scan.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Mamba2's state space recurrence for one head over a block of rows: each
// row's step and decay 32 rows to a vector, then each state row
// vectorized along the state dimension. Neither core has an f32 vector
// multiply, so each f32 product is summed from exact bf16 limb products
// (f32_split.h) and rounded once, as the scalar f32 product is.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/exp_f32.h"
#include "../common/f32_split.h"
#include "softplus.h"

#ifndef SCAN_ROWS
#define SCAN_ROWS 16
#endif
#ifndef SCAN_HEAD_DIM
#define SCAN_HEAD_DIM 16
#endif
#ifndef SCAN_BLOCKS
#define SCAN_BLOCKS 1
#endif
#ifndef SCAN_STATE
#define SCAN_STATE 32
#endif

namespace {

constexpr int P = SCAN_HEAD_DIM, N = SCAN_STATE;
constexpr unsigned L = N % 32 == 0 ? 32 : 16;
static_assert(P % 16 == 0 && N % 16 == 0,
              "head_dim and state must be whole 16-lane vectors");

template <unsigned W>
using acc_t = aie::accum<accfloat, W>;
template <unsigned W>
using bf16_t = aie::vector<bfloat16, W>;

alignas(aie::vector_decl_align) float s[P * N];
constexpr int RW = (SCAN_ROWS + 31) / 32 * 32;

alignas(aie::vector_decl_align) float ydot[P];
alignas(aie::vector_decl_align) float halves[16 * 16];
alignas(aie::vector_decl_align) bfloat16 dts[RW];
alignas(aie::vector_decl_align) float steps[RW], decays[RW];

// aie::reduce_add's pairwise sums for 16 rows at once: h holds each row's
// 16 lanes (for 32, the sums of lanes i and i + 16), and every later level
// pairs the same lanes across packed rows, so each total rounds as
// reduce_add's does.
static inline __attribute__((always_inline)) aie::vector<float, 16>
reduce_rows(const float *__restrict h) {
  using v32 = aie::vector<float, 32>;
  auto sum = [](v32 a, v32 b) {
    return aie::add(acc_t<32>(a), acc_t<32>(b)).template to_vector<float>();
  };
  v32 x[4], y[2];
  for (int m = 0; m < 4; m++) {
    const auto [lo, hi] = aie::interleave_unzip(
        aie::load_v<32>(h + 64 * m), aie::load_v<32>(h + 64 * m + 32), 8);
    x[m] = sum(lo, hi);
  }
  for (int j = 0; j < 2; j++) {
    const auto [lo, hi] = aie::interleave_unzip(x[2 * j], x[2 * j + 1], 4);
    y[j] = sum(lo, hi);
  }
  const auto [zl, zh] = aie::interleave_unzip(y[0], y[1], 2);
  const v32 z = sum(zl, zh);
  const auto [rl, rh] = aie::interleave_unzip(z, z, 1);
  return sum(rl, rh).template extract<16>(0);
}

} // namespace

// block is x (SCAN_ROWS x SCAN_HEAD_DIM), then bc (SCAN_ROWS rows of the
// group's B then C, SCAN_STATE each), then dt (SCAN_ROWS). head is
// (dt_bias, A, D), then the state before the head's first block. b counts
// blocks, SCAN_BLOCKS to a head, and block b loads the state when it is a
// head's first. Each block runs its first valid rows: step = softplus(dt +
// dt_bias), state = exp(step * A) * state + step * outer(x, B), y = state @ C +
// D * x. The rows after valid leave the state alone and write zeros. The state
// stays in the core from block to block; final is it after this block.
extern "C" void selective_scan(bfloat16 *restrict block, float *restrict head,
                               float *restrict final, bfloat16 *restrict y,
                               int32_t valid, int32_t b) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *x = block;
  const bfloat16 *bc = x + SCAN_ROWS * P;
  const bfloat16 *dt = bc + SCAN_ROWS * 2 * N;
  if (b % SCAN_BLOCKS == 0) {
    const float *h = head + 3;
    float *to = s;
    for (int i = 0; i < P * N / L; i++, h += L, to += L)
      aie::store_v(to, aie::load_unaligned_v<L>(h));
  }
  const Limbs<32> A = bf16_limbs(acc_t<32>(aie::broadcast<float, 32>(head[1])));
  const Limbs<16> d = bf16_limbs(acc_t<16>(aie::broadcast<float, 16>(head[2])));
  for (int t = 0; t < SCAN_ROWS; t++)
    dts[t] = dt[t];
  for (int t = 0; t < SCAN_ROWS; t += 32) {
    acc_t<32> z;
    z.from_vector(aie::load_v<32>(dts + t));
    const acc_t<32> step = softplus_f32(
        aie::add(z, acc_t<32>(aie::broadcast<float, 32>(head[0]))));
    aie::store_v(steps + t, step.to_vector<float>());
    aie::store_v(decays + t,
                 exp_f32(mul_limbs(bf16_limbs(step), A)).to_vector<float>());
  }
  for (int t = 0; t < SCAN_ROWS; t++) {
    bfloat16 *yt = y + t * P;
    if (t >= valid) {
      for (int p = 0; p < P; p += 16)
        aie::store_v(yt + p, aie::zeros<bfloat16, 16>());
      continue;
    }
    const bfloat16 *xt = x + t * P;
    const bfloat16 *B = bc + t * 2 * N;
    const bfloat16 *C = B + N;
    const Limbs<L> decay =
        bf16_limbs(acc_t<L>(aie::broadcast<float, L>(decays[t])));
    const Limbs<16> step =
        bf16_limbs(acc_t<16>(aie::broadcast<float, 16>(steps[t])));
    for (int p0 = 0; p0 < P; p0 += 16) {
      const Limbs<16> sx =
          bf16_limbs(mul_limbs(step, aie::load_v<16>(xt + p0)));
      for (int q = 0; q < 16; q++) {
        const Limbs<L> e = {aie::broadcast<bfloat16, L>(sx.hi[q]),
                            aie::broadcast<bfloat16, L>(sx.mid[q]),
                            aie::broadcast<bfloat16, L>(sx.lo[q])};
        float *sp = s + (p0 + q) * N;
        acc_t<L> dot = aie::zeros<accfloat, L>();
        AIE_LOOP_UNROLL_FULL
        for (int n = 0; n < N; n += L) {
          const bf16_t<L> Bn = aie::load_v<L>(B + n);
          const bf16_t<L> Cn = aie::load_v<L>(C + n);
          const Limbs<L> old = bf16_limbs(acc_t<L>(aie::load_v<L>(sp + n)));
          const acc_t<L> next =
              aie::add(mul_limbs(decay, old), mul_limbs(e, Bn));
          aie::store_v(sp + n, next.template to_vector<float>());
          dot = aie::add(dot, mul_limbs(bf16_limbs(next), Cn));
        }
        if constexpr (L == 32)
          aie::store_v(halves + q * 16, aie::add(dot.template extract<16>(0),
                                                 dot.template extract<16>(1))
                                            .template to_vector<float>());
        else
          aie::store_v(halves + q * 16, dot.template to_vector<float>());
      }
      aie::store_v(ydot + p0, reduce_rows(halves));
    }
    for (int p = 0; p < P; p += 16) {
      const acc_t<16> yp = aie::add(acc_t<16>(aie::load_v<16>(ydot + p)),
                                    mul_limbs(d, aie::load_v<16>(xt + p)));
      aie::store_v(yt + p, yp.to_vector<bfloat16>());
    }
  }
  const float *from = s;
  float *to = final;
  for (int i = 0; i < P * N / L; i++, from += L, to += L)
    aie::store_v(to, aie::load_v<L>(from));
  aie::set_rounding(saved_rounding);
  event1();
}
