//===- llama_rmsnorm_int8.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RMSNorm: the fp32 residual in, int8 out with its dynamic scale. AIE2P DMA
// cannot type-convert, so the requant lives inside the kernel.
//
//   inv_rms  = invsqrt(mean(x^2) + eps)
//   combined = inv_rms * inv_act_scale_out
//   y_i8     = clamp(round(x * combined * gamma[i]), -128, 127)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif

static constexpr float kEps = 1e-5f;
// One 512-bit fp32 register.
static constexpr int kLanes = 16;

// 1/sqrt(a) from the Quake-III seed and two Newton steps, as
// llama_oracle.sw_invsqrt mirrors it (not the correctly rounded value).
static inline float sw_invsqrt(float a) {
  int32_t bits = __builtin_bit_cast(int32_t, a);
  bits = (int32_t)0x5f3759df - (bits >> 1);
  float x = __builtin_bit_cast(float, bits);
  x = scalar_mul_rne(
      x, 1.5f - scalar_mul_rne(scalar_mul_rne(scalar_mul_rne(0.5f, a), x), x));
  x = scalar_mul_rne(
      x, 1.5f - scalar_mul_rne(scalar_mul_rne(scalar_mul_rne(0.5f, a), x), x));
  return x;
}

// sum(x^2) in fp32: each lane sums its squares in order, then the lanes are
// added in order 0..15 as scalars, the order llama_oracle.sum_sq_f32 takes.
static inline float vec_sum_sq_f32(const float *restrict x, int cols) {
  ::aie::vector<float, kLanes> acc = ::aie::zeros<float, kLanes>();
  for (int c = 0; c < cols / kLanes; c++) {
    ::aie::vector<float, kLanes> xv = ::aie::load_v<kLanes>(x + c * kLanes);
    acc = ::aie::add(acc, vec_mul_rne(xv, xv));
  }
  alignas(64) float lanes[kLanes];
  ::aie::store_v(lanes, acc);
  float sum = lanes[0];
  for (int l = 1; l < kLanes; l++)
    sum += lanes[l];
  return sum;
}

// (x * combined) * gamma[i], kLanes at a time: an exact widening (bf16->f32)
// and two products each rounded once, as the scalar `*` chain does per lane.
static inline ::aie::vector<float, kLanes>
vec_normalize_lanes(const float *restrict x, const bfloat16 *restrict gamma,
                    ::aie::vector<float, kLanes> combined_v) {
  ::aie::accum<accfloat, kLanes> g_acc;
  g_acc.from_vector(::aie::load_v<kLanes>(gamma));
  return vec_mul_rne(vec_mul_rne(::aie::load_v<kLanes>(x), combined_v),
                     g_acc.template to_vector<float>());
}

// absmax((x * pre) * gamma)'s bits, an integer max of the magnitudes' bits.
static inline int32_t vec_absmax_bits(const float *restrict x,
                                      const bfloat16 *restrict gamma,
                                      ::aie::vector<float, kLanes> pre_v,
                                      int cols) {
  ::aie::vector<int32, kLanes> mag = ::aie::zeros<int32, kLanes>();
  for (int c = 0; c < cols / kLanes; c++)
    mag = ::aie::max(
        mag, ::aie::bit_and(
                 int32(0x7fffffff),
                 vec_normalize_lanes(x + c * kLanes, gamma + c * kLanes, pre_v)
                     .template cast_to<int32>()));
  return ::aie::reduce_max(mag);
}

static inline void vec_requant(const float *restrict x,
                               const bfloat16 *restrict gamma,
                               int8_t *restrict y,
                               ::aie::vector<float, kLanes> combined_v,
                               int cols) {
  for (int c = 0; c < cols / kLanes; c++)
    ::aie::store_v(y + c * kLanes,
                   vec_round_to_i8(vec_normalize_lanes(
                       x + c * kLanes, gamma + c * kLanes, combined_v)));
}

// The norm's fast form, 32 lanes at a time, for a row whose products the
// guards in rmsnorm_f32 keep normal and finite: each is rounded as in
// x_bf16_mul (see there), srs by d, or d + 1 where it reaches 2^(d + 24). A
// lane whose x or gamma has exponent field 0 gets 0: its products' sign
// cannot reach the int8 row or the absmax. x's and gamma's
// nonzero exponent fields are tracked (ExpRange).

// sum(x^2) as vec_sum_sq_f32 adds it; x's exponent range into xr.
static inline float sum_sq_fast(const int32_t *restrict x, ExpRange &xr) {
  ::aie::vector<float, kLanes> acc = ::aie::zeros<float, kLanes>();
  ::aie::accum<acc64, 32> init;
  init.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 24);
  for (int c = 0; c < LLAMA_RMSNORM_COLS / 32; c++) {
    ::aie::vector<int32, 32> xb = ::aie::load_v<32>(x + 32 * c);
    XPlanes xp = x_planes(xb);
    exp_range_add(xr, xp.h);
    ::aie::accum<acc64, 32> p(sig24_mac<true>(xp.sig, xp.sig, init));
    ::aie::vector<int32, 32> lo = p.template to_vector<int32>(23);
    ::aie::vector<int32, 32> hi = p.template to_vector<int32>(24);
    ::aie::vector<int32, 32> e = ::aie::bit_and(int32(0x7f800000), xb);
    ::aie::vector<int32, 32> sq =
        ::aie::add(::aie::add(::aie::add(e, e), int32(-(126 << 23))),
                   ::aie::select(lo, hi, ::aie::ge(lo, int32(0))));
    sq =
        ::aie::select(sq, ::aie::zeros<int32, 32>(), ::aie::eq(xp.h, int16(0)));
    acc = ::aie::add(acc, sq.extract<16>(0).cast_to<float>());
    acc = ::aie::add(acc, sq.extract<16>(1).cast_to<float>());
  }
  alignas(64) float lanes[kLanes];
  ::aie::store_v(lanes, acc);
  float sum = lanes[0];
  for (int l = 1; l < kLanes; l++)
    sum += lanes[l];
  return sum;
}

// (x * a) * gamma's bits, 32 lanes, where a's significand planes are a_sig
// and its sign and exponent bits are folded into a0 and a1 (=
// (a & mask) - (253 << 23), and that plus 2^23), x and gamma masked alike
// (0x7f800000 for magnitudes, 0xff800000 for signed values).
template <int32 kMask>
static inline ::aie::vector<int32, 32>
scale_gamma_fast(::aie::vector<int32, 32> xb, ::aie::vector<int16, 32> gv,
                 Sig24 a_sig, int32 a0, int32 a1,
                 const ::aie::accum<acc64, 32> &init_g) {
  XPlanes xp = x_planes(xb);
  ::aie::accum<acc64, 32> p(sig24_mac<false>(xp.sig, a_sig, v32acc64()));
  ::aie::vector<int32, 32> lo = p.template to_vector<int32>(23);
  ::aie::vector<int32, 32> hi = p.template to_vector<int32>(24);
  auto wide = ::aie::ge(lo, int32(1 << 24));
  ::aie::vector<int32, 32> e =
      ::aie::add(::aie::bit_and(int32(kMask), xb), ::aie::select(a0, a1, wide));
  ::aie::vector<int32, 16> gw = gv.cast_to<int32>();
  auto [g_lo, g_hi] = ::aie::interleave_zip(::aie::zeros<int16, 32>(), gv, 1);
  ::aie::vector<int32, 32> g32 =
      ::aie::concat(g_lo.cast_to<int32>(), g_hi.cast_to<int32>());
  ::aie::vector<int16, 32> gh =
      ::aie::bit_and(int32(0x7f807f80), gw).cast_to<int16>();
  v32int16 gs =
      ::aie::bit_or(int32(0x00800080), ::aie::bit_and(int32(0x007f007f), gw))
          .cast_to<int16>();
  p = ::aie::accum<acc64, 32>(
      sig24x8_mac(sig24_planes(::aie::select(lo, hi, wide)), gs, init_g));
  lo = p.template to_vector<int32>(7);
  hi = p.template to_vector<int32>(8);
  ::aie::vector<int32, 32> v =
      ::aie::add(::aie::add(e, ::aie::bit_and(int32(kMask), g32)),
                 ::aie::select(lo, hi, ::aie::ge(lo, int32(0))));
  return ::aie::select(v, ::aie::zeros<int32, 32>(),
                       ::aie::eq(xp.h, int16(0)) | ::aie::eq(gh, int16(0)));
}

static inline int32_t absmax_fast(const int32_t *restrict x,
                                  const int16_t *restrict gamma, float a,
                                  const ::aie::accum<acc64, 32> &init_g) {
  const int32_t ab = __builtin_bit_cast(int32_t, a);
  const Sig24 a_sig =
      sig24_planes(::aie::broadcast<int32, 32>((ab & 0x7fffff) | 0x800000));
  const int32 a0 = (ab & 0x7f800000) - (253 << 23);
  ::aie::vector<int32, 32> mag = ::aie::zeros<int32, 32>();
  for (int c = 0; c < LLAMA_RMSNORM_COLS / 32; c++)
    mag = ::aie::max(
        mag, scale_gamma_fast<0x7f800000>(::aie::load_v<32>(x + 32 * c),
                                          ::aie::load_v<32>(gamma + 32 * c),
                                          a_sig, a0, a0 + (1 << 23), init_g));
  return ::aie::reduce_max(mag);
}

static inline void requant_fast(const int32_t *restrict x,
                                const int16_t *restrict gamma, float a,
                                const ::aie::accum<acc64, 32> &init_g,
                                int8_t *restrict y) {
  const int32_t ab = __builtin_bit_cast(int32_t, a);
  const Sig24 a_sig =
      sig24_planes(::aie::broadcast<int32, 32>((ab & 0x7fffff) | 0x800000));
  const int32 a0 = (int32_t)(ab & 0xff800000) - (253 << 23);
  for (int c = 0; c < LLAMA_RMSNORM_COLS / 32; c++)
    ::aie::store_v(
        y + 32 * c,
        vec_round_to_i8_bits(scale_gamma_fast<int32(0xff800000)>(
            ::aie::load_v<32>(x + 32 * c), ::aie::load_v<32>(gamma + 32 * c),
            a_sig, a0, a0 + (1 << 23), init_g)));
}

// The norm of the fp32 residual x[kCols] with gamma bf16[kCols] into y:
// int8[kCols + 8], the normed row, then its fp32 scale absmax/127 and 4 zero
// bytes.
static inline void rmsnorm_f32(const float *restrict x,
                               const bfloat16 *restrict gamma,
                               int8_t *restrict y) {
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kLanes == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 16");

  const int32_t *xb = reinterpret_cast<const int32_t *>(x);
  const int16_t *gb = reinterpret_cast<const int16_t *>(gamma);
  ExpRange xr = exp_range_empty(), gr = exp_range_empty();
  for (int c = 0; c < kCols / 32; c++)
    exp_range_add(
        gr,
        ::aie::bit_and(int32(0x7f807f80),
                       ::aie::load_v<32>(gb + 32 * c).template cast_to<int32>())
            .template cast_to<int16>());
  float sum_sq = sum_sq_fast(xb, xr);
  int x_lo, x_hi, g_lo, g_hi;
  exp_range_get(xr, x_lo, x_hi);
  exp_range_get(gr, g_lo, g_hi);
  const bool any = x_lo <= x_hi && g_lo <= g_hi;
  // Every product of the fast forms, x * x and (x * a) * gamma, normal:
  // exponent fields in [2, 253] before rounding (no flush, no overflow).
  if (x_lo <= x_hi && !(2 * x_lo - 127 >= 2 && 2 * x_hi - 126 <= 253))
    sum_sq = vec_sum_sq_f32(x, kCols);
  auto chain_ok = [&](float a) {
    const int e = __builtin_bit_cast(int32_t, a) >> 23;
    const int lo = x_lo + e - 127, hi = x_hi + e - 126;
    return e >= 1 && e <= 254 &&
           (!any || (lo >= 2 && hi <= 253 && lo + g_lo - 127 >= 2 &&
                     hi + g_hi - 126 <= 253));
  };
  ::aie::accum<acc64, 32> init_g;
  init_g.from_vector(::aie::broadcast<int32, 32>(-(1 << 23)), 8);

  float var = scalar_mul_rne(sum_sq, 1.0f / kCols);
  float inv_rms = sw_invsqrt(var + kEps);

  const int32_t absmax_bits =
      chain_ok(inv_rms)
          ? absmax_fast(xb, gb, inv_rms, init_g)
          : vec_absmax_bits(x, gamma, ::aie::broadcast<float, kLanes>(inv_rms),
                            kCols);
  const float absmax =
      __builtin_bit_cast(float, std::max(absmax_bits, kAbsmaxFloorBits));
  float scale_dyn = scalar_mul_rne(absmax, 1.0f / 127.0f);
  float combined = scalar_mul_rne(inv_rms, recip_rne(scale_dyn));
  if (chain_ok(combined))
    requant_fast(xb, gb, combined, init_g, y);
  else
    vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  __builtin_memcpy(y + kCols, &scale_dyn, 4);
  int32_t zero = 0;
  __builtin_memcpy(y + kCols + 4, &zero, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
}

extern "C" {

// xg: the fp32 residual x[kCols], then gamma bf16[kCols] (the decode stages
// the two together), normed into y as rmsnorm_f32.
void llama_rmsnorm_f32_packed(int8_t *restrict xg, int8_t *restrict y) {
  event0();
  rmsnorm_f32(reinterpret_cast<const float *>(xg),
              reinterpret_cast<const bfloat16 *>(xg + 4 * LLAMA_RMSNORM_COLS),
              y);
  event1();
}

// The prefill's: x and gamma (bf16, then padding) in buffers of their own.
void llama_rmsnorm_f32(int8_t *restrict x, int8_t *restrict gamma,
                       int8_t *restrict y) {
  event0();
  rmsnorm_f32(reinterpret_cast<const float *>(x),
              reinterpret_cast<const bfloat16 *>(gamma), y);
  event1();
}

// llama_rmsnorm_f32_packed, then x on to row: the residual the residual core
// adds the unit's output to.
void llama_rmsnorm_f32_packed_fwd(int8_t *restrict xg, int8_t *restrict y,
                                  int8_t *restrict row) {
  llama_rmsnorm_f32_packed(xg, y);
  for (int c = 0; c < 4 * LLAMA_RMSNORM_COLS / 64; c++)
    ::aie::store_v(row + c * 64, ::aie::load_v<64>(xg + c * 64));
}

} // extern "C"
