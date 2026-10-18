//===- llama_rope_int8.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama-3 RoPE on int8 activations with bf16 cos/sin LUTs. HALF-SPLIT
// rotation (HF llama convention), not interleaved:
//   x1, x2 = x[:d/2], x[d/2:]
//   out[:d/2] = x1*cos[:d/2] - x2*sin[:d/2];  out[d/2:] = x2*cos[d/2:] +
//   x1*sin[d/2:]
// Same scale in/out: rope is norm-preserving per (i, i+d/2) pair, so the
// kernel dequants int8->bf16, rotates, requants with the same scale.
// x is (n_heads, head_dim) flattened; cos/sin (head_dim,) reused per head.
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
  // Half-away-from-zero rounding implemented in INTEGER math to be
  // deterministic regardless of inherited fp rounding mode or 1-ULP
  // noise. Trick: compute s*((int)(s*v*2 + 1)/2) where s = sign.
  //   For v = -52.5:   s=-1, s*v*2 = 105.0, +1 = 106.0, (int)/2 = 53, *s = -53.
  //   (matches numpy half-away) For v = -52.499: s=-1, s*v*2 = 104.998, +1 =
  //   105.998, (int)/2 = 52, *s = -52.
  // The integer truncation in `(int32_t)(scaled) / 2` is what gives us
  // the half-away semantics deterministically -- (int) cast on a value
  // that ROUNDED to an integer-plus-one is independent of fp rounding
  // mode because the .5 boundary is doubled to a .0 (integer) value.
  int32_t sign = (v >= 0.0f) ? 1 : -1;
  float scaled = (float)sign * v * 2.0f + 1.0f;
  int32_t doubled = (int32_t)scaled;
  int32_t r = sign * (doubled / 2);
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

// Software fp32 reciprocal (see llama_flowkv.cc for why Peano's `/` needs
// this workaround).
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

// o1 = x1*c1 - x2*s1, o2 = x2*c2 + x1*s2, kHalf lanes at a time (one call
// covers a whole head's half). bf16->f32 widens are exact; the products
// accumulate in accfloat and round to f32 once, same discipline as
// aie_kernels/datamovement/rope.cc's rope_halves_step.
template <int kHalf>
static inline void
vec_rope_halves(::aie::vector<float, kHalf> x1f,
                ::aie::vector<float, kHalf> x2f, const bfloat16 *restrict cos,
                const bfloat16 *restrict sin, ::aie::vector<float, kHalf> &o1,
                ::aie::vector<float, kHalf> &o2) {
  ::aie::vector<bfloat16, kHalf> c1v = ::aie::load_v<kHalf>(cos);
  ::aie::vector<bfloat16, kHalf> s1v = ::aie::load_v<kHalf>(sin);
  ::aie::vector<bfloat16, kHalf> c2v = ::aie::load_v<kHalf>(cos + kHalf);
  ::aie::vector<bfloat16, kHalf> s2v = ::aie::load_v<kHalf>(sin + kHalf);

  ::aie::accum<accfloat, kHalf> c1a, s1a, c2a, s2a;
  c1a.from_vector(c1v);
  s1a.from_vector(s1v);
  c2a.from_vector(c2v);
  s2a.from_vector(s2v);
  ::aie::vector<float, kHalf> c1f = c1a.template to_vector<float>();
  ::aie::vector<float, kHalf> s1f = s1a.template to_vector<float>();
  ::aie::vector<float, kHalf> c2f = c2a.template to_vector<float>();
  ::aie::vector<float, kHalf> s2f = s2a.template to_vector<float>();

  o1 = ::aie::msc(::aie::mul(x1f, c1f), x2f, s1f).template to_vector<float>();
  o2 = ::aie::mac(::aie::mul(x2f, c2f), x1f, s2f).template to_vector<float>();
}

extern "C" {

// cs_packed layout: cos[head_dim] || sin[head_dim] -- packed into one
// bfloat16 buffer to fit the CT 2-in/2-out DMA channel budget.
void llama_rope_int8(int8_t *restrict x, bfloat16 *restrict cs_packed,
                     int8_t *restrict out, float act_scale) {
  event0();

  constexpr int kHD = LLAMA_ROPE_HEAD_DIM;
  constexpr int kNHeads = LLAMA_ROPE_N_HEADS;
  constexpr int kHalf = kHD / 2;

  static_assert(kHD % 2 == 0, "head_dim must be even");

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;
  const float inv_scale = sw_recip(act_scale);
  ::aie::vector<float, kHalf> scale_v =
      ::aie::broadcast<float, kHalf>(act_scale);
  ::aie::vector<float, kHalf> inv_scale_v =
      ::aie::broadcast<float, kHalf>(inv_scale);

  for (int h = 0; h < kNHeads; h++) {
    int base = h * kHD;
    ::aie::vector<int8, kHalf> x1v = ::aie::load_v<kHalf>(x + base);
    ::aie::vector<int8, kHalf> x2v = ::aie::load_v<kHalf>(x + base + kHalf);
    ::aie::vector<float, kHalf> x1f =
        ::aie::mul(::aie::to_float<float>(x1v), scale_v)
            .template to_vector<float>();
    ::aie::vector<float, kHalf> x2f =
        ::aie::mul(::aie::to_float<float>(x2v), scale_v)
            .template to_vector<float>();

    ::aie::vector<float, kHalf> o1, o2;
    vec_rope_halves<kHalf>(x1f, x2f, cos, sin, o1, o2);

    ::aie::vector<float, kHalf> o1s =
        ::aie::mul(o1, inv_scale_v).template to_vector<float>();
    ::aie::vector<float, kHalf> o2s =
        ::aie::mul(o2, inv_scale_v).template to_vector<float>();
    for (int i = 0; i < kHalf; i++) {
      out[base + i] = round_to_i8(o1s[i]);
      out[base + kHalf + i] = round_to_i8(o2s[i]);
    }
  }

  event1();
}

// Dynamic-scale variant (Phase 6c.5b.3). The `x` buffer is sized
// kHD*kNHeads + 8 bytes; the trailing 8 bytes carry (act_scale fp32,
// spare fp32) written upstream by q_proj's _perchan_v2_up variant. We
// read act_scale from there and additionally PASSTHROUGH the 8 B tail
// from x to out so flowkv_qk can pick up q_scale from the same offset.
void llama_rope_int8_dyn(int8_t *restrict x, bfloat16 *restrict cs_packed,
                         int8_t *restrict out) {
  event0();

  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);

  constexpr int kHD = LLAMA_ROPE_HEAD_DIM;
  constexpr int kNHeads = LLAMA_ROPE_N_HEADS;
  constexpr int kHalf = kHD / 2;
  constexpr int kBody = kHD * kNHeads;

  static_assert(kHD % 2 == 0, "head_dim must be even");

  // act_scale read only for tail passthrough (downstream still sees
  // q_scale); not used in the math -- rope is norm-preserving so
  // act_scale cancels, and including it reintroduced ULP-noise fp32
  // multiplies that moved a rounding boundary off numpy's.
  // Math: ((x*ac)*c - (x'*ac)*s) * (1/ac) ≡ x*c - x'*s   (real arith).
  // In fp32 the simplified form has fewer rounding steps and matches
  // numpy_rope when numpy is updated to the same simplified form.

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;

  for (int h = 0; h < kNHeads; h++) {
    int base = h * kHD;
    ::aie::vector<int8, kHalf> x1v = ::aie::load_v<kHalf>(x + base);
    ::aie::vector<int8, kHalf> x2v = ::aie::load_v<kHalf>(x + base + kHalf);
    ::aie::vector<float, kHalf> x1f = ::aie::to_float<float>(x1v);
    ::aie::vector<float, kHalf> x2f = ::aie::to_float<float>(x2v);

    ::aie::vector<float, kHalf> o1, o2;
    vec_rope_halves<kHalf>(x1f, x2f, cos, sin, o1, o2);

    for (int i = 0; i < kHalf; i++) {
      out[base + i] = round_to_i8(o1[i]);
      out[base + kHalf + i] = round_to_i8(o2[i]);
    }
  }

  // Passthrough the 8 B tail so downstream (flowkv_qk_dyn) reads the
  // same q_scale from out[kBody..kBody+8].
  memcpy(out + kBody, x + kBody, 8);

  event1();
}

} // extern "C"
