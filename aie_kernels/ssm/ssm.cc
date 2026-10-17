//===- ssm.cc ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mamba-2's selective state space scan for a unit of a head's state, S[n][p]:
// a prompt chunk's blocks (ssd_cb, ssd_decay, ssd_out, ssd_state) and a
// decode step (ssm_state_update). Every float32 product is taken from bf16
// limbs (f32_split.h).
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/exp2f.h"
#include "../common/f32_split.h"
#include "../common/log_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SSM_Q
#define SSM_Q 32
#endif
#ifndef SSM_P
#define SSM_P 32
#endif
#ifndef SSM_N
#define SSM_N 128
#endif
#ifndef SSM_DT
#define SSM_DT 64
#endif

constexpr int Q = SSM_Q;
constexpr int P = SSM_P;
constexpr int N = SSM_N;
constexpr int DT = SSM_DT;
static_assert(Q == f32_lanes, "a block is one 32-lane vector of positions");
static_assert(P == 16 || P == 32, "a state row is one 16- or 32-lane vector");
static_assert(N % f32_lanes == 0, "the state is whole 32-lane vectors");

using vf = aie::vector<float, f32_lanes>;
using vi = aie::vector<int32_t, f32_lanes>;
using p_acc = aie::accum<accfloat, P>;
using p_bf16 = aie::vector<bfloat16, P>;

// log2(e) and 1 as bf16 limbs.
static inline Limbs<f32_lanes> log2e() {
  return {bf16_splat(0x3fb9), bf16_splat(0xbb2c), bf16_splat(0x36ed)};
}

static inline f32_acc zero_acc() {
  return f32_acc(aie::zeros<float, f32_lanes>());
}

// -a, exactly, from a's limbs. A float32 negation by its sign bit (or
// aie::neg) does not legalize where it feeds a limb split (Peano 83ba9ea9:
// G_FNEG <32 x s32> on AIE2P, G_OR <8 x s64> on AIE2).
static inline f32_acc neg_f32(f32_acc a) {
  const Limbs<f32_lanes> l = bf16_limbs(a);
  const bf16_vec one = bf16_splat(0x3f80);
  return aie::msc(aie::msc(aie::msc(zero_acc(), l.hi, one), l.mid, one), l.lo,
                  one);
}

// a * b, each an f32 accumulator, to float32's accuracy.
static inline f32_acc mul_f32(f32_acc a, f32_acc b) {
  return mac_limbs(zero_acc(), bf16_limbs(a), bf16_limbs(b));
}

// e^x, lane by lane.
static inline f32_acc exp_f32(f32_acc x) {
  return f32_acc(exp2f_v(
      mac_limbs(zero_acc(), bf16_limbs(x), log2e()).to_vector<float>()));
}

// log(1 + e^x), lane by lane: max(x, 0) + log1p(e^-|x|), log1p of a t
// below sqrt(2) - 1 from t itself, which 1 + t would round.
static inline f32_acc softplus(f32_acc x) {
  // The f32 compares are emulated, so signs and order are read from the bits,
  // which order as integers among non-negative floats.
  const vi bits = x.to_vector<float>().cast_to<int32_t>();
  const aie::mask<f32_lanes> positive_x =
      aie::gt(bits, aie::zeros<int32_t, f32_lanes>());
  const vf minus_abs = aie::select(x.to_vector<float>(),
                                   neg_f32(x).to_vector<float>(), positive_x);
  const f32_acc t = exp_f32(f32_acc(minus_abs));
  const vf small = log_poly_add(zero_acc(), log_poly(t, bf16_splat(0x3f80)))
                       .to_vector<float>();
  const vf large = log_f32_acc(aie::add(t, 1.0f)).to_vector<float>();
  const vf l = aie::select(large, small,
                           aie::lt(t.to_vector<float>().cast_to<int32_t>(),
                                   aie::broadcast<int32_t, f32_lanes>(
                                       0x3ed413cd /* sqrt(2) - 1 */)));
  const vf positive = aie::select(aie::zeros<float, f32_lanes>(),
                                  x.to_vector<float>(), positive_x);
  return aie::add(f32_acc(l), positive);
}

// The lanes 0, 1, ..., 31.
static inline vi lanes() {
  alignas(aie::vector_decl_align) int32_t t[f32_lanes];
  for (int i = 0; i < f32_lanes; i++)
    t[i] = i;
  return aie::load_v<f32_lanes>(t);
}

// The limbs of a value at each position of a block.
struct Terms {
  alignas(aie::vector_decl_align) bfloat16 hi[f32_lanes], mid[f32_lanes],
      lo[f32_lanes];
};

// v's limbs into t, a 16-lane half at a time: AIE2 does not select the
// stores of a 32-lane accumulator's limbs (Peano 83ba9ea9).
static inline void store_limbs(Terms &t, f32_acc v) {
  const vf f = v.to_vector<float>();
  AIE_LOOP_UNROLL_FULL
  for (int h = 0; h < 2; h++) {
    const Limbs<16> l =
        bf16_limbs(aie::accum<accfloat, 16>(f.template extract<16>(h)));
    aie::store_v(t.hi + 16 * h, l.hi);
    aie::store_v(t.mid + 16 * h, l.mid);
    aie::store_v(t.lo + 16 * h, l.lo);
  }
}

// Which terms a lane l sums: from: a_0 .. a_l; after: a_{l+1} .. a_to.
enum class Lanes { from, after };

// sum_k a_k over the terms each lane takes, k up to to.
static f32_acc sum_terms(const Terms &a, Lanes which, int to) {
  const bf16_vec one = bf16_splat(0x3f80);
  const bf16_vec none = aie::zeros<bfloat16, f32_lanes>();
  const vi lane = lanes();
  f32_acc acc = zero_acc();
  AIE_LOOP_NO_UNROLL
  for (int k = 0; k <= to; k++) {
    const aie::mask<f32_lanes> into =
        which == Lanes::from ? aie::ge(lane, k) : aie::lt(lane, k);
    const bf16_vec on = aie::select(none, one, into);
    acc = aie::mac(acc, on, aie::broadcast<bfloat16, f32_lanes>(a.hi[k]));
    acc = aie::mac(acc, on, aie::broadcast<bfloat16, f32_lanes>(a.mid[k]));
    acc = aie::mac(acc, on, aie::broadcast<bfloat16, f32_lanes>(a.lo[k]));
  }
  return acc;
}

// dt = softplus(dt_raw + dt_bias) of each position's head column, 0 from
// position valid on; and A = -e^A_log.
static inline f32_acc block_dt(const bfloat16 *dt, const bfloat16 *dt_bias,
                               int32_t head, int32_t valid) {
  alignas(aie::vector_decl_align) bfloat16 raw[f32_lanes];
  for (int l = 0; l < Q; l++)
    raw[l] = dt[l * DT + head];
  const bf16_vec one = bf16_splat(0x3f80);
  f32_acc v = aie::mul(aie::load_v<f32_lanes>(raw), one);
  v = aie::mac(v, aie::broadcast<bfloat16, f32_lanes>(dt_bias[head]), one);
  const vf sp = softplus(v).to_vector<float>();
  return f32_acc(
      aie::select(aie::zeros<float, f32_lanes>(), sp, aie::lt(lanes(), valid)));
}

static inline f32_acc head_a(const bfloat16 *a_log, int32_t head) {
  const bf16_vec one = bf16_splat(0x3f80);
  const f32_acc e =
      exp_f32(aie::mul(aie::broadcast<bfloat16, f32_lanes>(a_log[head]), one));
  return neg_f32(e);
}

extern "C" {

// g[l][s] = sum_n c[l][n] b[s][n], float32: each product exact, summed in the
// accumulator.
void ssd_cb_bf16(const bfloat16 *__restrict c, const bfloat16 *__restrict b,
                 float *__restrict g) {
  event0();
  for (int l = 0; l < Q; l++) {
    for (int s = 0; s < Q; s++) {
      f32_acc acc = zero_acc();
      AIE_LOOP_UNROLL_FULL
      for (int n = 0; n < N; n += f32_lanes)
        acc = aie::mac(acc, aie::load_v<f32_lanes>(c + l * N + n),
                       aie::load_v<f32_lanes>(b + s * N + n));
      g[l * Q + s] = aie::reduce_add(acc.to_vector<float>());
    }
  }
  event1();
}

// A block of the chunked scan: ssd_decay once a head, then ssd_out and
// ssd_state a unit, with dt_l = softplus(dt[l][head] + dt_bias[head]) (0 from
// valid on) and a_l = -dt_l e^A_log[head]. Every sum of a is taken over its
// own terms, never as the difference of two longer ones.

// m[l][s] = g[l][s] e^{a_{s+1} + ... + a_l} dt_s for s <= l, else 0; and
// e = (e^{s_l}, e^{a_{l+1} + ... + a_{Q-1}} dt_l), each Q float32.
void ssd_decay_bf16(const float *__restrict g, const bfloat16 *__restrict dt,
                    const bfloat16 *__restrict a_log,
                    const bfloat16 *__restrict dt_bias, float *__restrict m,
                    float *__restrict e, int32_t head, int32_t valid) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const f32_acc dtv = block_dt(dt, dt_bias, head, valid);
  Terms a;
  store_limbs(a, mul_f32(dtv, head_a(a_log, head)));
  const vi lane = lanes();
  aie::store_v(e, exp_f32(sum_terms(a, Lanes::from, Q - 1)).to_vector<float>());
  aie::store_v(e + Q, mul_f32(exp_f32(sum_terms(a, Lanes::after, Q - 1)), dtv)
                          .to_vector<float>());
  for (int l = 0; l < Q; l++) {
    const vf decay =
        aie::select(aie::zeros<float, f32_lanes>(),
                    exp_f32(sum_terms(a, Lanes::after, l)).to_vector<float>(),
                    aie::le(lane, l));
    aie::store_v(m + l * Q,
                 mul_f32(mul_f32(f32_acc(aie::load_v<f32_lanes>(g + l * Q)),
                                 f32_acc(decay)),
                         dtv)
                     .to_vector<float>());
  }
  aie::set_rounding(saved);
  event1();
}

// y_l = sum_{s<=l} m[l][s] x_s + e^{s_l} sum_n c[l][n] S_in[n] + D[head] x_l,
// m and e from ssd_decay. The sums are an output's: m's and S_in's top two
// limbs serve, 2^-16 of each.
void ssd_out_bf16(const float *__restrict m, const float *__restrict e,
                  const bfloat16 *__restrict c, const bfloat16 *__restrict x,
                  const bfloat16 *__restrict d,
                  const float *__restrict state_in, bfloat16 *__restrict y,
                  int32_t head) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const p_bf16 p_one = aie::broadcast<bfloat16, P>(1.0f);
  for (int l = 0; l < Q; l++) {
    p_acc before = p_acc(aie::zeros<float, P>());
    for (int n = 0; n < N; n++) {
      const p_acc row = p_acc(aie::load_v<P>(state_in + n * P));
      p_bf16 hi, mid, lo;
      split3(row, hi, mid, lo, p_one);
      const p_bf16 cn = aie::broadcast<bfloat16, P>(c[l * N + n]);
      before = aie::mac(aie::mac(before, hi, cn), mid, cn);
    }
    Terms ml;
    store_limbs(ml, f32_acc(aie::load_v<f32_lanes>(m + l * Q)));
    p_acc within = p_acc(aie::zeros<float, P>());
    for (int t = 0; t <= l; t++) {
      const p_bf16 xt = aie::load_v<P>(x + t * P);
      within = aie::mac(within, xt, aie::broadcast<bfloat16, P>(ml.hi[t]));
      within = aie::mac(within, xt, aie::broadcast<bfloat16, P>(ml.mid[t]));
    }
    const Limbs<P> el = bf16_limbs(p_acc(aie::broadcast<float, P>(e[l])));
    p_acc out = mac_limbs(within, bf16_limbs(before), el);
    out = aie::mac(out, aie::load_v<P>(x + l * P),
                   aie::broadcast<bfloat16, P>(d[head]));
    aie::store_v(y + l * P, out.to_vector<bfloat16>());
  }
  aie::set_rounding(saved);
  event1();
}

// S_out = e^{s_{Q-1}} S_in + sum_l w_l b_l^T x_l, w = e[Q:] from ssd_decay;
// state_out may be state_in itself.
void ssd_state_bf16(const float *__restrict e, const bfloat16 *__restrict b,
                    const bfloat16 *__restrict x, const float *state_in,
                    float *state_out) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const Limbs<P> el = bf16_limbs(p_acc(aie::broadcast<float, P>(e[Q - 1])));
  for (int n = 0; n < N; n++) {
    const p_acc row = p_acc(aie::load_v<P>(state_in + n * P));
    aie::store_v(state_out + n * P,
                 mac_limbs(p_acc(aie::zeros<float, P>()), bf16_limbs(row), el)
                     .to_vector<float>());
  }
  for (int l = 0; l < Q; l++) {
    const Limbs<P> wl = bf16_limbs(p_acc(aie::broadcast<float, P>(e[Q + l])));
    const p_bf16 xl = aie::load_v<P>(x + l * P);
    p_acc u = aie::mul(xl, wl.hi);
    u = aie::mac(aie::mac(u, xl, wl.mid), xl, wl.lo);
    const Limbs<P> ul = bf16_limbs(u);
    for (int n = 0; n < N; n++) {
      const p_bf16 bn = aie::broadcast<bfloat16, P>(b[l * N + n]);
      p_acc acc = p_acc(aie::load_v<P>(state_out + n * P));
      acc = aie::mac(aie::mac(aie::mac(acc, ul.hi, bn), ul.mid, bn), ul.lo, bn);
      aie::store_v(state_out + n * P, acc.to_vector<float>());
    }
  }
  aie::set_rounding(saved);
  event1();
}

// One position of the recurrence for one unit, from state_in into state_out,
// which may be state_in itself:
//   dt = softplus(dt[head] + dt_bias[head]),  A = -e^A_log[head]
//   S_out = e^{dt A} S_in + b^T (dt x)
//   y     = sum_n c[n] S_out[n] + D[head] x
void ssm_state_update_bf16(
    const bfloat16 *__restrict x, const bfloat16 *__restrict b,
    const bfloat16 *__restrict c, const bfloat16 *__restrict dt,
    const bfloat16 *__restrict a_log, const bfloat16 *__restrict d,
    const bfloat16 *__restrict dt_bias, const float *state_in, float *state_out,
    bfloat16 *__restrict y, int32_t head) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec one = bf16_splat(0x3f80);
  f32_acc v = aie::mul(aie::broadcast<bfloat16, f32_lanes>(dt[head]), one);
  v = aie::mac(v, aie::broadcast<bfloat16, f32_lanes>(dt_bias[head]), one);
  // dt and e^{dt A} in every lane.
  const f32_acc dtv = softplus(v);
  const f32_acc decay_v = exp_f32(mul_f32(dtv, head_a(a_log, head)));
  const Limbs<P> decay =
      bf16_limbs(p_acc(decay_v.to_vector<float>().template extract<P>(0)));
  const Limbs<P> dl =
      bf16_limbs(p_acc(dtv.to_vector<float>().template extract<P>(0)));
  const p_bf16 xv = aie::load_v<P>(x);
  p_acc u = aie::mul(xv, dl.hi);
  u = aie::mac(aie::mac(u, xv, dl.mid), xv, dl.lo);
  const Limbs<P> ul = bf16_limbs(u);
  p_acc out = p_acc(aie::zeros<float, P>());
  for (int n = 0; n < N; n++) {
    const p_acc row = p_acc(aie::load_v<P>(state_in + n * P));
    p_acc next =
        mac_limbs(p_acc(aie::zeros<float, P>()), bf16_limbs(row), decay);
    const p_bf16 bn = aie::broadcast<bfloat16, P>(b[n]);
    next = aie::mac(aie::mac(aie::mac(next, ul.hi, bn), ul.mid, bn), ul.lo, bn);
    aie::store_v(state_out + n * P, next.to_vector<float>());
    const Limbs<P> nl = bf16_limbs(next);
    const p_bf16 cn = aie::broadcast<bfloat16, P>(c[n]);
    out = aie::mac(aie::mac(aie::mac(out, nl.hi, cn), nl.mid, cn), nl.lo, cn);
  }
  out = aie::mac(out, xv, aie::broadcast<bfloat16, P>(d[head]));
  aie::store_v(y, out.to_vector<bfloat16>());
  aie::set_rounding(saved);
  event1();
}

} // extern "C"
