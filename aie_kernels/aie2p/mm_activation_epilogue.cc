//===- mm_activation_epilogue.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fused, runtime-mode-selected GEMM epilogue for aie2p: given the f32
// accumulator tile a matmul microkernel produces, apply one of
// {identity, SiLU, GELU} and write f32 out -- selected per DISPATCH by a
// single runtime-parameter word rather than by which xclbin is loaded.
//
//   out = act(acc),  acc f32 -> out f32,  act in {identity, SiLU, GELU}
//
// Because the activation is per-element, this epilogue does not care what
// tiled/blocked coordinate order the producing matmul's C tile is stored
// in -- unlike a reduction-shaped epilogue (LayerNorm, say), the caller
// can run this directly on the raw accumulator; de-shuffling to row-major,
// if needed at all, can happen downstream of this kernel instead of
// before it.
//
// No bias argument: a common way to fold a per-output-column bias into a
// GEMM without a second on-chip op is to K-augment the matmul itself (one
// extra reduction step, A-operand column = 1, B-operand row = bias), so by
// the time this epilogue runs the bias is already summed into `acc`. That
// is a caller-side choice this kernel does not require -- it is a pure
// activation-only epilogue; a caller without that trick can add its own
// bias before calling, or pass an unbiased `acc` and use mode 0 (identity)
// as a plain matmul output.
//
// Mode selection (the `mode` argument, 0/1/2 below): the same compiled
// kernel body branches at runtime rather than requiring a separate xclbin
// per activation, so one resident program can serve identity/SiLU/GELU
// dispatches back-to-back with no reconfiguration between them, driven by
// an RTP (run-time parameter) write on the caller's side -- see the
// accompanying programming_examples/ml/mm_activation_epilogue example for the
// IRON-side `Buffer(..., use_write_rtp=True)` wiring.
//
// sigmoid is synthesized from the tanh SFU (there is no sigmoid SFU on
// this core):
//   sigmoid(z) = 1/(1+e^-z) = 0.5*(1 + tanh(z/2))
//
// PRECISION (SiLU): three variants of "SiLU via bf16 tanh" were tried
// before settling on the one below. All-bf16 (narrow the accumulator to
// bf16 first, do the whole computation there) is fastest but rounds the
// accumulator twice -- once before the tanh argument, once before the
// final multiply -- which measurably lost accuracy against an f32
// reference in my own pipeline. All-f32 (accumulate, tanh, and multiply
// all kept in f32) recovers that accuracy but hangs on device: f32
// elementwise/transcendental ops are not free on this bf16-native unit,
// and the extra width blew the per-tile cycle budget in my testing. The
// hybrid below keeps x and the FINAL x*sigmoid multiply in f32 (so the
// accumulator is only rounded once, inside the tanh itself) and narrows
// only the sigmoid, which is bounded to [0, 1] and so tolerates bf16
// well -- one extra f32 multiply over the all-bf16 path, no hang, and no
// measurable accuracy loss against an f32 reference in my testing.
//
// GELU reuses the same bf16 tanh approximation strategy as the all-bf16
// SiLU path (not the hybrid): an f32 polynomial GELU (cube + tanh kept in
// f32) was also tried and hit the same on-device hang as all-f32 SiLU, for
// the same reason -- so it stays bf16 throughout, narrowed once at input.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// out = acc, per-element copy (no activation). Runtime n a multiple of 16.
static inline void mm_identity_row(uint32_t n, const float *__restrict acc,
                                   float *__restrict out) {
  event0();
  for (uint32_t off = 0; off < n; off += 16) {
    aie::store_v(out + off, aie::load_v<16>(acc + off));
  }
  event1();
}

// SiLU: out = x * sigmoid(x), hybrid precision -- see the file header.
// Runtime n a multiple of 16.
static inline void mm_silu_hiprec_row(uint32_t n, const float *__restrict acc,
                                      float *__restrict out) {
  event0();
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);
  for (uint32_t off = 0; off < n; off += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(acc + off);
    // Keep x/2 (the tanh argument) in f32; only the tanh OUTPUT narrows to
    // bf16 -- it is bounded to [-1, 1], so bf16 is accurate enough there.
    aie::vector<float, 16> half_x = aie::mul(x, halff);
    aie::vector<bfloat16, 16> tanh_half_x = aie::tanh<bfloat16>(half_x);
    aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb); // bf16, in [0,1]
    // Up-convert sigmoid to f32; the final multiply uses the UN-rounded x.
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
// bf16 throughout after the initial narrow -- see the file header for why
// this does not use the SiLU hybrid's f32 final multiply. Runtime n a
// multiple of 16.
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

// mode: 0 = identity, 1 = SiLU (hiprec), 2 = GELU. `n` a multiple of 16.
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
