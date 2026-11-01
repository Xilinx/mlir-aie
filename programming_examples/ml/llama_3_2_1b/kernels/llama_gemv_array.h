//===- llama_gemv_array.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// What the GEMV array's kernels (llama_gemv_array.cc, llama_gemv_mma.cc) and
// the prefill's (llama_gemm_mma.cc, llama_prefill_glue.cc) share:
// llama_vec_fp.h's fp32 helpers, the mma epilogue and the Q rope.
//===----------------------------------------------------------------------===//

#ifndef LLAMA_GEMV_ARRAY_H
#define LLAMA_GEMV_ARRAY_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

// A KV group's 4 Q heads (q: 4 x 64 fp32) roped in fp32 with a position's
// cos/sin (cs_bits: bf16 cos[64], then sin[64]) as llama_kv_row.h ropes k,
// then requantized per head into out: 256 int8, then per head [scale f32 |
// 0]. Needs rounding_mode::conv_even and saturation_mode::saturate.
static inline void rope_q_requant4(const float *restrict q,
                                   const uint16_t *restrict cs_bits,
                                   int8_t *restrict out) {
  constexpr int kHD = 64;
  constexpr int kHalf = kHD / 2;
  alignas(64) float cs_f[2 * kHD];
  for (int i = 0; i < 2 * kHD; i++)
    cs_f[i] = __builtin_bit_cast(float, (uint32_t)cs_bits[i] << 16);
  alignas(64) float roped[4 * kHD];
  for (int h = 0; h < 4; h++)
    for (int j = 0; j < kHalf; j += 16) {
      ::aie::vector<float, 16> x1 = ::aie::load_v<16>(q + h * kHD + j);
      ::aie::vector<float, 16> x2 = ::aie::load_v<16>(q + h * kHD + kHalf + j);
      ::aie::store_v(
          roped + h * kHD + j,
          ::aie::sub(vec_mul_rne(x1, ::aie::load_v<16>(cs_f + j)),
                     vec_mul_rne(x2, ::aie::load_v<16>(cs_f + kHD + j))));
      ::aie::store_v(
          roped + h * kHD + kHalf + j,
          ::aie::add(
              vec_mul_rne(x2, ::aie::load_v<16>(cs_f + kHalf + j)),
              vec_mul_rne(x1, ::aie::load_v<16>(cs_f + kHD + kHalf + j))));
    }
  q_requant_heads<4, kHD>(roped, out, out + 256);
}

// The exponent fields of the (positive) scales the epilogue's vector form
// takes.
static constexpr uint32_t kEpilogueExpLo = 80, kEpilogueExpHi = 174;

// A scale's significand with its implicit bit, for epilogue_rne.
static inline Sig24 epilogue_sig(::aie::vector<int32, 32> bits) {
  return sig24_planes(
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bits)));
}

// A scale's sign and exponent bits less `bias` << 23, for epilogue_rne.
static inline ::aie::vector<int32, 32>
epilogue_exp(::aie::vector<int32, 32> bits, int32 bias) {
  return ::aie::sub(::aie::bit_and(int32(0xff800000), bits), int32(bias << 23));
}

// epilogue_rne's state between its stages: the stage's exact product, the
// result's sign and exponent bits so far and the lanes with s = 0.
struct EpilogueState {
  ::aie::accum<acc64, 32> p;
  ::aie::vector<int32, 32> e;
  ::aie::mask<32> zero;
};

static inline EpilogueState epilogue_first(::aie::vector<int32, 32> s, Sig24 am,
                                           uint32_t &wide_s) {
  ::aie::vector<int32, 32> xs =
      ::aie::to_float<float>(s, 0).template cast_to<int32>();
  Sig24 x = sig24_planes(xs);
  // See x_planes.
  ::aie::vector<int32, 16> x_hi =
      ::aie::vector<int16, 32>(x.hi).cast_to<int32>();
  ::aie::vector<int16, 32> h =
      ::aie::bit_and(int32(0x7fff7fff), x_hi).cast_to<int16>();
  wide_s |= ::aie::ge(h, int16(150 << 7)).to_uint32();
  x.hi =
      ::aie::bit_or(int32(0x00800080), ::aie::bit_and(int32(0x007f007f), x_hi))
          .cast_to<int16>();
  return {::aie::accum<acc64, 32>(sig24_mac<false>(x, am, v32acc64())),
          ::aie::bit_and(int32(0xff800000), xs), ::aie::eq(h, int16(0))};
}

static inline EpilogueState epilogue_second(const EpilogueState &x, Sig24 wm,
                                            ::aie::vector<int32, 32> ae,
                                            ::aie::vector<int32, 32> ae1,
                                            ::aie::vector<int32, 32> we) {
  ::aie::vector<int32, 32> lo = x.p.template to_vector<int32>(23);
  ::aie::vector<int32, 32> hi = x.p.template to_vector<int32>(24);
  auto wide = ::aie::ge(lo, int32(1 << 24));
  Sig24 m = sig24_planes(::aie::select(lo, hi, wide));
  ::aie::accum<acc64, 32> init;
  init.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 24);
  return {::aie::accum<acc64, 32>(sig24_mac<true>(m, wm, init)),
          ::aie::add(::aie::add(x.e, ::aie::select(ae, ae1, wide)), we),
          x.zero};
}

static inline ::aie::vector<float, 32> epilogue_third(const EpilogueState &x) {
  ::aie::vector<int32, 32> lo = x.p.template to_vector<int32>(23);
  ::aie::vector<int32, 32> hi = x.p.template to_vector<int32>(24);
  return ::aie::select(
             ::aie::add(x.e, ::aie::select(lo, hi, ::aie::ge(lo, int32(0)))),
             ::aie::zeros<int32, 32>(), x.zero)
      .template cast_to<float>();
}

// Lane-wise RNE(RNE(float(s) * a) * w), as the scalar `*` rounds it, for
// positive a, w with exponent fields in [kEpilogueExpLo, kEpilogueExpHi], on
// the lanes with s in (-2^23, 2^23); sets bit j of `wide_s` for lane j
// otherwise (float(s)'s exponent is at least 150 however it rounds). Every
// product is then normal and no rounded significand reaches 2^24, so each is
// rounded as in x_bf16_mul (see there): srs by 23, or 24 where the product
// reaches 2^47 (for the second, where the product less 2^47 rounds to >= 0).
// am, wm:
// epilogue_sig of a, w; ae, ae1: epilogue_exp(a, 126), epilogue_exp(a, 125);
// we: epilogue_exp(w, 127). Needs rounding_mode::conv_even.
static inline ::aie::vector<float, 32>
epilogue_rne(::aie::vector<int32, 32> s, Sig24 am, Sig24 wm,
             ::aie::vector<int32, 32> ae, ::aie::vector<int32, 32> ae1,
             ::aie::vector<int32, 32> we, uint32_t &wide_s) {
  return epilogue_third(
      epilogue_second(epilogue_first(s, am, wide_s), wm, ae, ae1, we));
}

// The f32 epilogue's vector form on activations 4h..4h+3's 32 sums (lane
// (m - 4h)*8 + n, act_scales[m - 4h]): adds the 8 rows' biases to sums and,
// where epilogue_rne's guards hold for every lane (ok), returns the 32
// values. A weight slot's trailer holds the 8 rows' int32 biases, then their
// fp32 scales.
static inline ::aie::vector<float, 32>
half_epilogue_vec(::aie::vector<int32, 32> &sums,
                  const float *restrict act_scales,
                  const int8_t *restrict trailer, bool &ok) {
  const int32_t *bias = reinterpret_cast<const int32_t *>(trailer);
  const float *w_scales = reinterpret_cast<const float *>(trailer + 32);
  ok = true;
  for (int i = 0; i < 4 + 8; i++) {
    float f = i < 4 ? act_scales[i] : w_scales[i - 4];
    ok &= (__builtin_bit_cast(uint32_t, f) >> 23) - kEpilogueExpLo <=
          kEpilogueExpHi - kEpilogueExpLo;
  }
  ::aie::vector<int32, 8> w8 = ::aie::load_v<8>(w_scales).cast_to<int32>();
  ::aie::vector<int32, 32> w_v = ::aie::concat(w8, w8, w8, w8);
  ::aie::vector<int32, 32> a_v =
      ::aie::concat(::aie::broadcast<float, 8>(act_scales[0]),
                    ::aie::broadcast<float, 8>(act_scales[1]),
                    ::aie::broadcast<float, 8>(act_scales[2]),
                    ::aie::broadcast<float, 8>(act_scales[3]))
          .cast_to<int32>();
  ::aie::vector<int32, 8> b8 = ::aie::load_v<8>(bias);
  sums = ::aie::add(sums, ::aie::concat(b8, b8, b8, b8));
  uint32_t wide_s = 0;
  ::aie::vector<float, 32> fp = epilogue_rne(
      sums, epilogue_sig(a_v), epilogue_sig(w_v), epilogue_exp(a_v, 126),
      epilogue_exp(a_v, 125), epilogue_exp(w_v, 127), wide_s);
  ok &= wide_s == 0;
  return fp;
}

// The f32 epilogue on activations 4h..4h+3's 32 sums, bias added, to
// out[m*ostride + row0 + n] for the first m_count activations: on 32 lanes
// when half_epilogue_vec's guards hold, else per element.
__attribute__((noinline)) static void
half_epilogue(::aie::vector<int32, 32> sums, const float *restrict act_scales,
              const int8_t *restrict trailer, int8_t *restrict out, int h,
              int32_t ostride, int32_t row0, int32_t m_count) {
  const float *w_scales = reinterpret_cast<const float *>(trailer + 32);
  const int m_end = m_count - 4 * h < 4 ? m_count - 4 * h : 4;
  bool ok;
  ::aie::vector<float, 32> fp =
      half_epilogue_vec(sums, act_scales, trailer, ok);
  for (int m = 0; m < m_end; m++)
    for (int n = 0; n < 8; n++) {
      const int lane = m * 8 + n;
      float out_val;
      if (ok) {
        out_val = fp.get(lane);
      } else {
        out_val = scalar_mul_rne(
            scalar_mul_rne((float)sums.get(lane), act_scales[m]), w_scales[n]);
      }
      reinterpret_cast<float *>(out)[(4 * h + m) * ostride + row0 + n] =
          out_val;
    }
}

#endif // LLAMA_GEMV_ARRAY_H
