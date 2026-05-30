//===- llama_rmsnorm.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama-3-style RMSNorm on bf16 with per-element gamma.
//
//   y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
//
// Adapted from aie_kernels/aie2p/rms_norm.cc (which hard-coded scalar
// gamma = 1.0) to take a gamma vector input. First real Llama kernel
// landed on hardware; primarily exercises:
//   - aie::mul_square + reduce-and-accumulate pass over the input
//   - aie::invsqrt scalar HW intrinsic
//   - per-element multiply (norm) + (gamma)
//
// Phase 2 simplifications relative to the production design:
//   - bf16 in/out (the production path is int8 with per-tensor act
//     scales; that wrapper goes on top of this once bf16 is bit-exact)
//   - no fused residual add yet (residual is a separate add Worker;
//     fusion is a future opt)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif
#ifndef LLAMA_RMSNORM_N
#define LLAMA_RMSNORM_N 16   // bf16 vector lane count
#endif

extern "C" {

void llama_rmsnorm_bf16(bfloat16 *restrict input, bfloat16 *restrict gamma,
                        bfloat16 *restrict output) {
  event0();

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  constexpr int kN = LLAMA_RMSNORM_N;
  constexpr int kChunks = kCols / kN;
  constexpr float kEps = 1e-5f;

  // Pass 1: sum of squares.
  ::aie::accum<acc32, kN> acc = ::aie::zeros<acc32, kN>();
  ::aie::vector<float, kN> add_res = ::aie::zeros<float, kN>();
  for (int i = 0; i < kChunks; i++) {
    ::aie::vector<bfloat16, kN> x = ::aie::load_v<kN>(input + i * kN);
    ::aie::vector<float, kN> sq = ::aie::mul_square(x);
    acc = ::aie::add(add_res, sq);
    add_res = acc.template to_vector<float>();
  }
  float sum_sq = ::aie::reduce_add(add_res);

  // Pass 2: normalize and scale by gamma.
  float rms = sum_sq / kCols + kEps;
  float inv_rms = aie::invsqrt(rms);
  ::aie::vector<bfloat16, kN> inv_rms_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(inv_rms));

  for (int i = 0; i < kChunks; i++) {
    ::aie::vector<bfloat16, kN> x = ::aie::load_v<kN>(input + i * kN);
    ::aie::vector<bfloat16, kN> g = ::aie::load_v<kN>(gamma + i * kN);
    ::aie::vector<bfloat16, kN> norm = ::aie::mul(x, inv_rms_v);
    ::aie::vector<bfloat16, kN> out  = ::aie::mul(norm, g);
    ::aie::store_v(output + i * kN, out);
  }

  event1();
}

} // extern "C"
