//===- llama_silu_mul_int8.cc -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama silu_mul: fused SiLU(gate) * up. Production dataflow shape
// (int8 in, int8 out; bf16 internal for the transcendental).
//
//   gate_f = gate_i8 * gate_scale
//   up_f   = up_i8   * up_scale
//   silu_g = gate_f * sigmoid(gate_f)
//   out_f  = silu_g * up_f
//   out_i8 = clamp(round(out_f * inv_out_scale), -128, 127)
//
// SiLU uses the aie::tanh<bfloat16> -> sigmoid identity from
// aie_kernels/aie2p/silu.cc:
//   sigmoid(x) = (tanh(x/2) + 1) / 2
//
//===----------------------------------------------------------------------===//

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_SILU_MUL_COLS
#define LLAMA_SILU_MUL_COLS 8192   // HIDDEN_DIM
#endif
#ifndef LLAMA_SILU_MUL_N
#define LLAMA_SILU_MUL_N 16
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX) r = I8_MAX;
  if (r < I8_MIN) r = I8_MIN;
  return (int8_t)r;
}

extern "C" {

void llama_silu_mul_int8(int8_t *restrict gate, int8_t *restrict up,
                         int8_t *restrict out,
                         float gate_scale, float up_scale,
                         float inv_out_scale) {
  event0();

  constexpr int kCols = LLAMA_SILU_MUL_COLS;
  constexpr int kN = LLAMA_SILU_MUL_N;
  constexpr int kChunks = kCols / kN;

  ::aie::vector<bfloat16, kN> gate_scale_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(gate_scale));
  ::aie::vector<bfloat16, kN> up_scale_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(up_scale));
  ::aie::vector<bfloat16, kN> half_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(0.5f));
  ::aie::vector<bfloat16, kN> one_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(1.0f));

  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int i = 0; i < kChunks; i++) {
    // Dequant gate and up.
    ::aie::vector<int8, kN> gi = ::aie::load_v<kN>(gate + i * kN);
    ::aie::vector<int8, kN> ui = ::aie::load_v<kN>(up   + i * kN);
    ::aie::vector<bfloat16, kN> gb = ::aie::to_float<bfloat16>(gi);
    ::aie::vector<bfloat16, kN> ub = ::aie::to_float<bfloat16>(ui);
    ::aie::vector<bfloat16, kN> gf = ::aie::mul(gb, gate_scale_v).template to_vector<bfloat16>();
    ::aie::vector<bfloat16, kN> uf = ::aie::mul(ub, up_scale_v).template to_vector<bfloat16>();

    // SiLU via tanh: sigmoid(x) = (tanh(x/2) + 1) / 2; silu = x * sigmoid.
    // aie::mul on bf16 returns an accum; .to_vector<float>() gives the
    // fp32 view that aie::tanh<bfloat16> wants (matches silu.cc usage).
    auto half_x_acc = ::aie::mul(gf, half_v);
    ::aie::vector<bfloat16, kN> tanh_half =
        ::aie::tanh<bfloat16>(half_x_acc.template to_vector<float>());
    ::aie::vector<bfloat16, kN> sig_plus_1 = ::aie::add(tanh_half, one_v);
    ::aie::vector<bfloat16, kN> sigmoid =
        ::aie::mul(sig_plus_1, half_v).template to_vector<bfloat16>();
    ::aie::vector<bfloat16, kN> silu =
        ::aie::mul(gf, sigmoid).template to_vector<bfloat16>();

    ::aie::vector<bfloat16, kN> out_bf =
        ::aie::mul(silu, uf).template to_vector<bfloat16>();

    // Requant: scalar round-and-clamp to int8 (vec round-and-narrow is
    // a follow-up; matches rmsnorm_int8's pattern).
    for (int j = 0; j < kN; j++) {
      float v = static_cast<float>(out_bf[j]) * inv_out_scale;
      out[i * kN + j] = round_to_i8(v);
    }
  }

  event1();
}

} // extern "C"
