//===- layer_norm_f32.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/scalar_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// f32 per-row LayerNorm, optionally with a per-column affine and a narrowing
// output cast. The bf16 layer_norm (../norm/) centers with a single
// E[x^2] - mean^2 reduction, which the bf16 input contract makes safe: a bf16
// value near a large mean has an ulp wider than the std, so that regime is
// unrepresentable. On f32 input the mean can be large relative to the std and
// E[x^2] - mean^2 catastrophically cancels, so this one takes the two-pass
// centered variance instead: center first, then square.
template <typename TIn, typename TOut, int N, bool kAffine>
static inline void layer_norm_f32_impl(const TIn *restrict input,
                                       TOut *restrict output,
                                       const TIn *restrict gamma,
                                       const TIn *restrict beta, int32_t cols) {
  static_assert(kAffine || std::is_same_v<TOut, TIn>,
                "the non-affine instantiation writes TIn straight through, so "
                "TOut must equal TIn");
  event0();
  constexpr float epsilon = 1e-5f;
  const int chunks = (uint32_t)cols / N;
  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));

  // Pass 1: mean = sum(x) / cols.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  if (chunks > 0) {
    const TIn *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < chunks; i++) {
      sum_acc = ::aie::add(sum_acc, ::aie::load_v<N>(p));
      p += N;
    }
  }
  float mean = scalar_mul(
      ::aie::reduce_add(sum_acc.template to_vector<float>()), inv_cols);
  ::aie::vector<TIn, N> mean_v = ::aie::broadcast<TIn, N>((TIn)mean);

  // Pass 2: variance = sum((x - mean)^2) / cols (centered two-pass).
  ::aie::accum<accfloat, N> var_acc = ::aie::zeros<accfloat, N>();
  if (chunks > 0) {
    const TIn *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < chunks; i++) {
      var_acc =
          ::aie::mac_square(var_acc, ::aie::sub(::aie::load_v<N>(p), mean_v));
      p += N;
    }
  }
  float variance = scalar_mul(
      ::aie::reduce_add(var_acc.template to_vector<float>()), inv_cols);
  float inv_std = scalar_invsqrt(variance + epsilon);
  ::aie::vector<TIn, N> inv_std_v = ::aie::broadcast<TIn, N>((TIn)inv_std);

  // The two instantiations diverge only in where gamma/beta come from and
  // whether the write narrows.
  if constexpr (kAffine) {
    // conv_even makes the narrowing write agree bit-for-bit with a host
    // f32 -> bf16 pack. The mode is one sticky register shared by every
    // kernel on this core, so it is handed back before returning.
    ::aie::rounding_mode saved_rounding =
        ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    if (chunks > 0) {
      const TIn *restrict pi = input;
      const TIn *restrict pg = gamma;
      const TIn *restrict pb = beta;
      TOut *restrict po = output;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<TIn, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
        ::aie::vector<TIn, N> norm_v =
            ::aie::mul(diff_v, inv_std_v).template to_vector<TIn>();
        // Kept as a separate multiply and add rather than a mac: an FMA would
        // skip the rounding of norm * gamma that the host reference performs.
        ::aie::vector<TIn, N> scaled_v =
            ::aie::mul(norm_v, ::aie::load_v<N>(pg)).template to_vector<TIn>();
        ::aie::vector<TIn, N> out_v =
            ::aie::add(scaled_v, ::aie::load_v<N>(pb));
        ::aie::accum<accfloat, N> a;
        a.from_vector(out_v);
        ::aie::store_v(po, a.template to_vector<TOut>());
        pi += N;
        pg += N;
        pb += N;
        po += N;
      }
    }
    ::aie::set_rounding(saved_rounding);
  } else {
    // gamma = 1 and beta = 0 here, so the affine pair is not applied at all.
    if (chunks > 0) {
      const TIn *restrict pi = input;
      TOut *restrict po = output;
      AIE_LOOP_UNROLL(2)
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<TIn, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
        ::aie::store_v(po,
                       ::aie::mul(diff_v, inv_std_v).template to_vector<TIn>());
        pi += N;
        po += N;
      }
    }
  }

  event1();
}

#if AIE_TUNED_AIE2
// aie_api's f32 multiply is emulated on AIE2, so here an f32 is split into
// bf16 limbs, two holding its top 16 bits and three all of it.
#include "../common/bf16_limbs.h"

static inline v16accfloat f32_acc(const float *restrict p) {
  return v16accfloat(v16float(::aie::load_v<16>(p)));
}

static inline v16accfloat f32_lanes(float x) {
  return broadcast_to_v16accfloat(x);
}

// 1 / sqrt(m) to f32, for a normal m > 0.
static inline v16accfloat inv_sqrt(v16accfloat m_acc) {
  v16accfloat m_rest;
  v32bfloat16 m = limbs(m_acc, m_rest);
  v16bfloat16 m2 = to_v16bfloat16(m_rest);
  v16bfloat16 zero = bf16_lanes(0);
  v32bfloat16 ones = broadcast_one_to_v32bfloat16();
  v32bfloat16 y2 = inv_sqrt_limbs(m_acc, m);

  // y2 is good to about 1e-5, and exact in its two limbs. One more step,
  // y2 + y2 r / 2 with r = 1 - (m y2) y2, where m y2 rounded to f32 leaves r
  // off by 2^-24.
  v16bfloat16 y2_0 = limb(y2, 0), y2_1 = limb(y2, 1);
  v16accfloat t_rest;
  v32bfloat16 t = limbs(
      mac_elem_16_2(bf16_pair(m2, zero), bf16_pair(y2_0, zero),
                    mac_elem_16_2(m, bf16_pair(y2_1, y2_1),
                                  mul_elem_16_2(m, bf16_pair(y2_0, y2_0)))),
      t_rest);
  v16accfloat r = msc_elem_16_2(t, bf16_pair(y2_0, y2_0), f32_lanes(1.0f));
  r = msc_elem_16_2(t, bf16_pair(y2_1, y2_1), r);
  r = msc_elem_16_2(bf16_pair(to_v16bfloat16(t_rest), zero),
                    bf16_pair(y2_0, zero), r);
  v16bfloat16 r_half = to_v16bfloat16(mul_elem_16_2(
      bf16_pair(to_v16bfloat16(r), zero), bf16_pair(bf16_lanes(0x3f00), zero)));
  return mac_elem_16_2(y2, bf16_pair(r_half, r_half), mul_elem_16_2(y2, ones));
}

// a / n to f32 for a >= 0: q = a inv_n to 16 bits, then one correction from
// the residual a - n q, which is exact since n and q are exact in two limbs.
static inline v16accfloat div_n(v16accfloat a, v32bfloat16 n,
                                v32bfloat16 inv_n) {
  v16bfloat16 n0 = limb(n, 0), n1 = limb(n, 1);
  v16bfloat16 i0 = limb(inv_n, 0), i1 = limb(inv_n, 1);
  v32bfloat16 a_l = limbs(a);
  v32bfloat16 q = limbs(mac_elem_16_2(a_l, bf16_pair(i1, i1),
                                      mul_elem_16_2(a_l, bf16_pair(i0, i0))));
  v16accfloat r = msc_elem_16_2(q, bf16_pair(n1, n1),
                                msc_elem_16_2(q, bf16_pair(n0, n0), a));
  v16bfloat16 r0 = to_v16bfloat16(r);
  return mac_elem_16_2(bf16_pair(r0, r0), inv_n,
                       mul_elem_16_2(q, broadcast_one_to_v32bfloat16()));
}

// x - mean for 16 lanes: [d0 | d1] and d2.
static inline v32bfloat16 centered(const float *restrict p, v16accfloat mean,
                                   v16bfloat16 &d2) {
  v16accfloat rest;
  v32bfloat16 d = limbs(sub(f32_acc(p), mean), rest);
  d2 = to_v16bfloat16(rest);
  return d;
}

// (x - mean) s to f32 with s = s0 + s1 + s2: the products of limbs i + j <= 2.
static inline void normalize_f32(const float *restrict in, float *restrict out,
                                 v16accfloat mean, v32bfloat16 s00,
                                 v32bfloat16 s11, v32bfloat16 s20) {
  v16bfloat16 d2;
  v32bfloat16 d = centered(in, mean, d2);
  v16accfloat y = mul_elem_16_2(d, s00);
  y = mac_elem_16_2(d, s11, y);
  y = mac_elem_16_2(bf16_pair(limb(d, 0), d2), s20, y);
  ::aie::store_v(out, ::aie::vector<float, 16>(v16float(y)));
}

// (x - mean) s gamma + beta to bf16, from three limbs of n and of gamma: the
// bf16 store rounds once, where the reference rounds norm * gamma, + beta, and
// the cast.
static inline void normalize_affine(const float *restrict in,
                                    const float *restrict gamma,
                                    const float *restrict beta,
                                    bfloat16 *restrict out, v16accfloat mean,
                                    v32bfloat16 s00, v32bfloat16 s11,
                                    v32bfloat16 s20) {
  v16bfloat16 d2;
  v32bfloat16 d = centered(in, mean, d2);
  v16accfloat n_rest, g_rest;
  v32bfloat16 n =
      limbs(mac_elem_16_2(bf16_pair(limb(d, 0), d2), s20,
                          mac_elem_16_2(d, s11, mul_elem_16_2(d, s00))),
            n_rest);
  v32bfloat16 g = limbs(f32_acc(gamma), g_rest);
  v16bfloat16 g0 = limb(g, 0), n0 = limb(n, 0);
  v16accfloat a = mac_elem_16_2(n, g, f32_acc(beta));
  a = mac_elem_16_2(bf16_pair(limb(n, 1), to_v16bfloat16(n_rest)),
                    bf16_pair(g0, g0), a);
  v16accfloat b = mul_elem_16_2(bf16_pair(n0, n0),
                                bf16_pair(limb(g, 1), to_v16bfloat16(g_rest)));
  ::aie::store_v(out, ::aie::vector<bfloat16, 16>(to_v16bfloat16(add(a, b))));
}

// layer_norm_f32_impl's three passes, with the multiplies in limbs.
template <typename TOut, bool kAffine>
static void layer_norm_f32_aie2(const float *restrict input,
                                TOut *restrict output,
                                const float *restrict gamma,
                                const float *restrict beta, int32_t cols) {
  event0();
  constexpr unsigned N = 16;
  const unsigned chunks = (uint32_t)cols / N;
  // See MIN_CHUNKS in ../norm/layer_norm_aie2.h.
  constexpr unsigned MIN_CHUNKS = 8;
  const bool pipelined = chunks >= MIN_CHUNKS;
  // 1 / n from a Q31 quotient. Divided first: no vector state is live across
  // the call.
  const int32_t inv_n_q31 = (int32_t)(0x80000000u / (uint32_t)cols);

  // Pass 1: sum(x), in four accumulators.
  v16accfloat sum0 = ::aie::zeros<accfloat, 16>();
  v16accfloat sum1 = ::aie::zeros<accfloat, 16>();
  v16accfloat sum2 = ::aie::zeros<accfloat, 16>();
  v16accfloat sum3 = ::aie::zeros<accfloat, 16>();
  unsigned chunk = 0;
  if (pipelined) {
    const float *restrict p = input;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS / 4)
    for (unsigned i = 0; i < chunks / 4; i++) {
      sum0 = add(sum0, f32_acc(p));
      sum1 = add(sum1, f32_acc(p + N));
      sum2 = add(sum2, f32_acc(p + 2 * N));
      sum3 = add(sum3, f32_acc(p + 3 * N));
      p += 4 * N;
    }
    chunk = chunks & ~3u;
  }
  for (; chunk < chunks; chunk++)
    sum0 = add(sum0, f32_acc(input + chunk * N));
  v16accfloat sum = f32_lanes(::aie::reduce_add(::aie::vector<float, 16>(
      v16float(add(add(sum0, sum1), add(sum2, sum3))))));

  v32bfloat16 n = limbs(f32_lanes(::aie::to_float<float>(cols, 0)));
  v32bfloat16 inv_n = limbs(f32_lanes(::aie::to_float<float>(inv_n_q31, 31)));
  v16accfloat mean = div_n(sum, n, inv_n);

  // Pass 2: sum((x - mean)^2) as d0^2 + d1^2 + 2 (d0 d1 + d0 d2) per element.
  // Only the totals are kept, so a mac may pair lanes of different products.
  v16accfloat sq = ::aie::zeros<accfloat, 16>();
  v16accfloat cross = ::aie::zeros<accfloat, 16>();
  chunk = 0;
  if (pipelined) {
    v16accfloat sq_b = ::aie::zeros<accfloat, 16>();
    v16accfloat cross_b = ::aie::zeros<accfloat, 16>();
    const float *restrict p = input;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS / 2)
    for (unsigned i = 0; i < chunks / 2; i++) {
      v16bfloat16 d2, e2;
      v32bfloat16 d = centered(p, mean, d2);
      v32bfloat16 e = centered(p + N, mean, e2);
      v16bfloat16 d0 = limb(d, 0), e0 = limb(e, 0);
      sq = mac_elem_16_2(d, d, sq);
      sq_b = mac_elem_16_2(e, e, sq_b);
      cross =
          mac_elem_16_2(bf16_pair(d0, d0), bf16_pair(limb(d, 1), d2), cross);
      cross_b =
          mac_elem_16_2(bf16_pair(e0, e0), bf16_pair(limb(e, 1), e2), cross_b);
      p += 2 * N;
    }
    sq = add(sq, sq_b);
    cross = add(cross, cross_b);
    chunk = chunks & ~1u;
  }
  {
    for (unsigned i = chunk; i < chunks; i++) {
      v16bfloat16 d2;
      v32bfloat16 d = centered(input + i * N, mean, d2);
      v16bfloat16 d0 = limb(d, 0);
      sq = mac_elem_16_2(d, d, sq);
      cross =
          mac_elem_16_2(bf16_pair(d0, d0), bf16_pair(limb(d, 1), d2), cross);
    }
  }
  v16accfloat q = f32_lanes(::aie::reduce_add(
      ::aie::vector<float, 16>(v16float(add(sq, add(cross, cross))))));
  v16accfloat s_rest;
  v32bfloat16 s =
      limbs(inv_sqrt(add(div_n(q, n, inv_n), f32_lanes(1e-5f))), s_rest);
  v16bfloat16 s0 = limb(s, 0), s1 = limb(s, 1), s2 = to_v16bfloat16(s_rest);
  v32bfloat16 s00 = bf16_pair(s0, s0), s11 = bf16_pair(s1, s1);

  // Pass 3.
  v32bfloat16 s20 = bf16_pair(s2, s0);
  if constexpr (kAffine) {
    if (pipelined) {
      const float *restrict pi = input;
      const float *restrict pg = gamma;
      const float *restrict pb = beta;
      bfloat16 *restrict po = output;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS)
      for (unsigned i = 0; i < chunks; i++) {
        normalize_affine(pi, pg, pb, po, mean, s00, s11, s20);
        pi += N;
        pg += N;
        pb += N;
        po += N;
      }
    } else {
      for (unsigned i = 0; i < chunks; i++)
        normalize_affine(input + i * N, gamma + i * N, beta + i * N,
                         output + i * N, mean, s00, s11, s20);
    }
  } else {
    if (pipelined) {
      const float *restrict pi = input;
      float *restrict po = output;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(MIN_CHUNKS / 4)
      for (unsigned i = 0; i < chunks / 4; i++) {
        normalize_f32(pi, po, mean, s00, s11, s20);
        normalize_f32(pi + N, po + N, mean, s00, s11, s20);
        normalize_f32(pi + 2 * N, po + 2 * N, mean, s00, s11, s20);
        normalize_f32(pi + 3 * N, po + 3 * N, mean, s00, s11, s20);
        pi += 4 * N;
        po += 4 * N;
      }
      for (unsigned i = 0; i < (chunks & 3); i++)
        normalize_f32(pi + i * N, po + i * N, mean, s00, s11, s20);
    } else {
      for (unsigned i = 0; i < chunks; i++)
        normalize_f32(input + i * N, output + i * N, mean, s00, s11, s20);
    }
  }
  event1();
}
#elif AIE_TUNED_AIE2P
// aie_api's f32 vector multiply is emulated on AIE2P as well: all nine
// products of three bf16 limbs, added one at a time, 16 lanes per call. Here
// the limbs are taken once per operand, only the six products of limbs
// i + j <= 2 are summed, and each mac covers 64 lanes.
constexpr unsigned kLanes = 64;
constexpr unsigned kPart = 16;
using f32_acc = ::aie::accum<accfloat, kLanes>;
using f32xN = ::aie::vector<float, kLanes>;
using bf16xN = ::aie::vector<bfloat16, kLanes>;

struct limbs3 {
  bf16xN l0, l1, l2;
};

static inline f32_acc to_acc(f32xN v) {
  f32_acc a;
  a.from_vector(v);
  return a;
}

// Three bf16 limbs summing to x exactly.
static inline limbs3 split3(f32_acc x) {
  const bf16xN one = ::aie::broadcast<bfloat16, kLanes>(1.0f);
  limbs3 r;
  r.l0 = x.to_vector<bfloat16>();
  x = ::aie::msc(x, r.l0, one);
  r.l1 = x.to_vector<bfloat16>();
  x = ::aie::msc(x, r.l1, one);
  r.l2 = x.to_vector<bfloat16>();
  return r;
}

// acc + x y, smallest products first.
static inline f32_acc mac3(f32_acc acc, const limbs3 &x, const limbs3 &y) {
  acc = ::aie::mac(acc, x.l2, y.l0);
  acc = ::aie::mac(acc, x.l1, y.l1);
  acc = ::aie::mac(acc, x.l0, y.l2);
  acc = ::aie::mac(acc, x.l1, y.l0);
  acc = ::aie::mac(acc, x.l0, y.l1);
  return ::aie::mac(acc, x.l0, y.l0);
}

static inline f32_acc mul3(const limbs3 &x, const limbs3 &y) {
  f32_acc acc = ::aie::mul(x.l2, y.l0);
  acc = ::aie::mac(acc, x.l1, y.l1);
  acc = ::aie::mac(acc, x.l0, y.l2);
  acc = ::aie::mac(acc, x.l1, y.l0);
  acc = ::aie::mac(acc, x.l0, y.l1);
  return ::aie::mac(acc, x.l0, y.l0);
}

// a b in every lane.
static inline f32_acc mul_bcast(float a, float b) {
  return mul3(split3(to_acc(::aie::broadcast<float, kLanes>(a))),
              split3(to_acc(::aie::broadcast<float, kLanes>(b))));
}

// (x - mean) s.
static inline f32xN normalize(f32xN x, f32_acc mean, const limbs3 &s) {
  return mul3(split3(::aie::sub(to_acc(x), mean)), s).to_vector<float>();
}

// (x - mean) s gamma + beta, rounded once to bf16.
static inline bf16xN normalize_affine(f32xN x, f32xN gamma, f32xN beta,
                                      f32_acc mean, const limbs3 &s) {
  limbs3 n = split3(mul3(split3(::aie::sub(to_acc(x), mean)), s));
  return mac3(to_acc(beta), n, split3(to_acc(gamma))).to_vector<bfloat16>();
}

// The last cols % 64 of a row, in 16-lane parts over a vector of pad.
static inline f32xN load_tail(const float *p, unsigned parts, float pad) {
  f32xN v = ::aie::broadcast<float, kLanes>(pad);
  v.insert(0, ::aie::load_v<kPart>(p));
  if (parts > 1)
    v.insert(1, ::aie::load_v<kPart>(p + kPart));
  if (parts > 2)
    v.insert(2, ::aie::load_v<kPart>(p + 2 * kPart));
  return v;
}

template <typename T>
static inline void store_tail(T *p, ::aie::vector<T, kLanes> v,
                              unsigned parts) {
  ::aie::store_v(p, v.template extract<kPart>(0));
  if (parts > 1)
    ::aie::store_v(p + kPart, v.template extract<kPart>(1));
  if (parts > 2)
    ::aie::store_v(p + 2 * kPart, v.template extract<kPart>(2));
}

// layer_norm_f32_impl's three passes. A row's last cols % 64 elements take
// one 64-lane step whose unused lanes are padded so that they add nothing:
// zeros for the sum, the mean for the variance.
template <typename TOut, bool kAffine>
static void layer_norm_f32_aie2p(const float *restrict input,
                                 TOut *restrict output,
                                 const float *restrict gamma,
                                 const float *restrict beta, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const unsigned chunks = (uint32_t)cols / kLanes;
  const unsigned tail = chunks * kLanes;
  const unsigned parts = ((uint32_t)cols - tail) / kPart;

  // Pass 1: mean = sum(x) / cols.
  f32_acc sum = ::aie::zeros<accfloat, kLanes>();
  if (chunks > 0) {
    const float *restrict p = input;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < chunks; i++) {
      sum = ::aie::add(sum, ::aie::load_v<kLanes>(p));
      p += kLanes;
    }
  }
  if (parts)
    sum = ::aie::add(sum, load_tail(input + tail, parts, 0.0f));
  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));
  const f32_acc mean_acc =
      mul_bcast(::aie::reduce_add(sum.to_vector<float>()), inv_cols);
  const float mean = mean_acc.to_vector<float>()[0];

  // Pass 2: sum((x - mean)^2) as d0^2 + d1^2 + 2 (d0 d1 + d0 d2) per element.
  f32_acc q = ::aie::zeros<accfloat, kLanes>();
  f32_acc c = ::aie::zeros<accfloat, kLanes>();
  if (chunks > 0) {
    const float *restrict p = input;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    AIE_LOOP_UNROLL(2)
    for (unsigned i = 0; i < chunks; i++) {
      limbs3 d = split3(::aie::sub(to_acc(::aie::load_v<kLanes>(p)), mean_acc));
      q = ::aie::mac(::aie::mac(q, d.l1, d.l1), d.l0, d.l0);
      c = ::aie::mac(::aie::mac(c, d.l0, d.l2), d.l0, d.l1);
      p += kLanes;
    }
  }
  if (parts) {
    limbs3 d = split3(
        ::aie::sub(to_acc(load_tail(input + tail, parts, mean)), mean_acc));
    q = ::aie::mac(::aie::mac(q, d.l1, d.l1), d.l0, d.l0);
    c = ::aie::mac(::aie::mac(c, d.l0, d.l2), d.l0, d.l1);
  }
  q = ::aie::add(q, ::aie::add(c, c));
  const float variance =
      mul_bcast(::aie::reduce_add(q.to_vector<float>()), inv_cols)
          .to_vector<float>()[0];
  const limbs3 s = split3(to_acc(
      ::aie::broadcast<float, kLanes>(scalar_invsqrt(variance + epsilon))));

  // Pass 3.
  if constexpr (kAffine) {
    if (chunks > 0) {
      const float *restrict pi = input;
      const float *restrict pg = gamma;
      const float *restrict pb = beta;
      TOut *restrict po = output;
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      AIE_LOOP_UNROLL(2)
      for (unsigned i = 0; i < chunks; i++) {
        ::aie::store_v(po, normalize_affine(::aie::load_v<kLanes>(pi),
                                            ::aie::load_v<kLanes>(pg),
                                            ::aie::load_v<kLanes>(pb), mean_acc,
                                            s));
        pi += kLanes;
        pg += kLanes;
        pb += kLanes;
        po += kLanes;
      }
    }
    if (parts)
      store_tail(output + tail,
                 normalize_affine(load_tail(input + tail, parts, mean),
                                  load_tail(gamma + tail, parts, 0.0f),
                                  load_tail(beta + tail, parts, 0.0f), mean_acc,
                                  s),
                 parts);
  } else {
    if (chunks > 0) {
      const float *restrict pi = input;
      TOut *restrict po = output;
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      AIE_LOOP_UNROLL(2)
      for (unsigned i = 0; i < chunks; i++) {
        ::aie::store_v(po, normalize(::aie::load_v<kLanes>(pi), mean_acc, s));
        pi += kLanes;
        po += kLanes;
      }
    }
    if (parts)
      store_tail(output + tail,
                 normalize(load_tail(input + tail, parts, mean), mean_acc, s),
                 parts);
  }
  event1();
}
#endif

extern "C" {
void layer_norm_f32(float *input, float *output, int32_t cols) {
#if AIE_TUNED_AIE2
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  layer_norm_f32_aie2<float, false>(input, output, nullptr, nullptr, cols);
  ::aie::set_rounding(saved_rounding);
#elif AIE_TUNED_AIE2P
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  layer_norm_f32_aie2p<float, false>(input, output, nullptr, nullptr, cols);
  ::aie::set_rounding(saved_rounding);
#else
  layer_norm_f32_impl<float, float, 16, false>(input, output, nullptr, nullptr,
                                               cols);
#endif
}

// LayerNorm + per-column affine + f32 -> bfloat16 cast in one dispatch. `gb`
// packs gamma then beta into one `[2 * cols]` buffer so that the kernel takes
// two DMA inputs, the AIE2p compute-tile limit; see `norm_affine` in
// programming_examples/ml/norm/norm.py for the matching packing.
void layer_norm_affine_cast(float *input, float *gb, bfloat16 *output,
                            int32_t cols) {
#if AIE_TUNED_AIE2
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  layer_norm_f32_aie2<bfloat16, true>(input, output, gb, gb + cols, cols);
  ::aie::set_rounding(saved_rounding);
#elif AIE_TUNED_AIE2P
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  layer_norm_f32_aie2p<bfloat16, true>(input, output, gb, gb + cols, cols);
  ::aie::set_rounding(saved_rounding);
#else
  layer_norm_f32_impl<float, bfloat16, 16, true>(input, output, gb, gb + cols,
                                                 cols);
#endif
}
}
