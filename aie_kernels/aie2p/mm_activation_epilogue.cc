//===- mm_activation_epilogue.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

static inline void mm_identity_row(uint32_t n, const float *__restrict acc,
                                   float *__restrict out) {
  event0();
  for (uint32_t off = 0; off < n; off += 16) {
    aie::store_v(out + off, aie::load_v<16>(acc + off));
  }
  event1();
}

// SiLU: out = x * sigmoid(x), sigmoid built from the tanh SFU as
// 0.5*(1 + tanh(x/2)). x and the final multiply stay f32, so the accumulator
// rounds once, inside the tanh; only sigmoid narrows, where its [0, 1] range
// is harmless. An all-f32 chain overruns the per-tile cycle budget and hangs.
static inline void mm_silu_hiprec_row(uint32_t n, const float *__restrict acc,
                                      float *__restrict out) {
  event0();
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);
  for (uint32_t off = 0; off < n; off += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(acc + off);
    aie::vector<float, 16> half_x = aie::mul(x, halff);
    aie::vector<bfloat16, 16> tanh_half_x = aie::tanh<bfloat16>(half_x);
    aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb);
    aie::accum<accfloat, 16> sacc;
    sacc.from_vector(sig);
    aie::vector<float, 16> sigf = sacc.to_vector<float>();
    aie::vector<float, 16> outv = aie::mul(x, sigf);
    aie::store_v(out + off, outv);
  }
  event1();
}

// GELU (tanh approximation, matches torch's gelu(approximate="tanh")):
//   gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
static inline void mm_gelu_row(uint32_t n, const float *__restrict acc,
                               float *__restrict out) {
  event0();
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> c0 =
      aie::broadcast<bfloat16, 16>(0.7978845608f); // sqrt(2/pi)
  const aie::vector<bfloat16, 16> c1 = aie::broadcast<bfloat16, 16>(0.044715f);
  for (uint32_t off = 0; off < n; off += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(acc + off));
    aie::vector<bfloat16, 16> x = a.to_vector<bfloat16>();
    aie::vector<bfloat16, 16> x2 = aie::mul(x, x);
    aie::vector<bfloat16, 16> x3 = aie::mul(x2, x);
    aie::vector<bfloat16, 16> c1x3 = aie::mul(c1, x3);
    aie::vector<bfloat16, 16> inner_b = aie::add(x, c1x3);
    auto inner = aie::mul(c0, inner_b);
    aie::vector<bfloat16, 16> t = aie::tanh<bfloat16>(inner.to_vector<float>());
    aie::vector<bfloat16, 16> t_p1 = aie::add(t, one);
    aie::vector<bfloat16, 16> xt = aie::mul(x, t_p1);
    aie::vector<bfloat16, 16> gx = aie::mul(half, xt);
    aie::accum<accfloat, 16> oacc;
    oacc.from_vector(gx);
    aie::store_v(out + off, oacc.to_vector<float>());
  }
  event1();
}

extern "C" {

// mode: 0 = identity, 1 = SiLU, 2 = GELU. `n` a multiple of 16.
void mm_activation_epilogue_row(const float *__restrict c_in,
                                float *__restrict c_out, int32_t n,
                                int32_t mode) {
  if (mode == 1) {
    mm_silu_hiprec_row((uint32_t)n, c_in, c_out);
  } else if (mode == 2) {
    mm_gelu_row((uint32_t)n, c_in, c_out);
  } else {
    mm_identity_row((uint32_t)n, c_in, c_out);
  }
}

} // extern "C"
