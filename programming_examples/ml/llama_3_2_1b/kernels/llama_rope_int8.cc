//===- llama_rope_int8.cc -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama-3 RoPE on int8 activations with bf16 cos/sin LUTs.
//
// Layout: HALF-SPLIT rotation (Llama-3 / HuggingFace convention), not
// interleaved (the layout aie_kernels/aie2p/rope.cc assumes). For
// head_dim d:
//   x1, x2 = x[:d/2], x[d/2:]
//   out[:d/2] = x1 * cos[:d/2]  -  x2 * sin[:d/2]
//   out[d/2:] = x2 * cos[d/2:]  +  x1 * sin[d/2:]
// (cos and sin are length d; their two halves are equal -- doubled for
// vector convenience -- per HF llama implementation.)
//
// Same scale for in and out: rope is norm-preserving per (i, i+d/2)
// pair, so |out| == |x| pointwise.  The kernel dequants int8 -> bf16,
// rotates, requants int8 using the same scale.
//
// Apply to all heads in one call: x layout is (n_heads, head_dim)
// flattened; cos/sin (head_dim,) are reused per head.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_ROPE_HEAD_DIM
#define LLAMA_ROPE_HEAD_DIM 64
#endif
#ifndef LLAMA_ROPE_N_HEADS
#define LLAMA_ROPE_N_HEADS 1
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX) r = I8_MAX;
  if (r < I8_MIN) r = I8_MIN;
  return (int8_t)r;
}

// Software fp32 reciprocal: Peano AIE2P lowers `1.0f / x` to a HW
// reciprocal approximation, NOT IEEE-correct. NR over a bit-hack
// initial guess converges to IEEE fp32 precision (4 iters).
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

extern "C" {

// cs_packed layout: cos[head_dim] || sin[head_dim] -- packed into one
// bfloat16 buffer to fit the CT 2-in/2-out DMA channel budget.
void llama_rope_int8(int8_t *restrict x, bfloat16 *restrict cs_packed,
                     int8_t *restrict out, float act_scale) {
  event0();

  constexpr int kHD     = LLAMA_ROPE_HEAD_DIM;
  constexpr int kNHeads = LLAMA_ROPE_N_HEADS;
  constexpr int kHalf   = kHD / 2;

  static_assert(kHD % 2 == 0, "head_dim must be even");

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;
  const float inv_scale = sw_recip(act_scale);

  // Scalar pair-wise rotation. Vec optimization is a follow-up; rope is
  // a small fraction of the layer wall, so scalar is acceptable for v0.
  for (int h = 0; h < kNHeads; h++) {
    int base = h * kHD;
    for (int i = 0; i < kHalf; i++) {
      float x1f = (float)x[base + i]          * act_scale;
      float x2f = (float)x[base + kHalf + i]  * act_scale;
      float c1  = (float)cos[i];
      float s1  = (float)sin[i];
      float c2  = (float)cos[kHalf + i];
      float s2  = (float)sin[kHalf + i];

      float o1 = x1f * c1 - x2f * s1;
      float o2 = x2f * c2 + x1f * s2;

      out[base + i]         = round_to_i8(o1 * inv_scale);
      out[base + kHalf + i] = round_to_i8(o2 * inv_scale);
    }
  }

  event1();
}

} // extern "C"
