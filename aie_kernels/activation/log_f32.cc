//===- log_f32.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifdef LOG_ELEMS
#define LOG_STAGED (!AIE_ARCH_AIE2 && LOG_ELEMS % 256 == 0)
#else
#define LOG_ELEMS n
#define LOG_STAGED 0
#endif

// log(m) for m in [sqrt(1/2), sqrt(2)) is f q(f) with f = m - 1, and q is
// log1p(f) / f interpolated at the 10 Chebyshev nodes of that range (numpy
// polyfit, degree 9): 2.5e-9 from log1p(f) with these float32 coefficients.
static const float log_q[10] = {1.0f,
                                -0.49999988079071045f,
                                0.33333346247673035f,
                                -0.2500157952308655f,
                                0.20000939071178436f,
                                -0.1660836935043335f,
                                0.14199650287628174f,
                                -0.13266031444072723f,
                                0.12806610763072968f,
                                -0.07451186329126358f};

// One Horner step of q: split q into limbs and return q f + c.
static inline __attribute__((always_inline)) f32_acc
log_step(f32_acc q, bf16_vec fh, bf16_vec fm, bf16_vec fl, float c,
         bf16_vec one) {
  bf16_vec qh, qm, ql;
  split3(q, qh, qm, ql, one);
  q = aie::mul(ql, fh);
  q = aie::mac(aie::mac(q, qm, fm), qh, fl);
  q = aie::mac(aie::mac(q, qm, fh), qh, fm);
  return aie::add(aie::mac(q, qh, fh), c);
}

#if LOG_STAGED
// Elements per staged block: eight trips a pass. Under the factory's
// eight-stage cap Peano miscompiled one-trip passes; eight measure exact.
static constexpr int LOG_BLOCK = 256;
// A block's q, f limbs and e, between passes.
// q ping-pongs: a pass that read and wrote one array would carry a store to
// the next iteration's load, which keeps the loop from pipelining.
alignas(64) static float log_q_block[2][LOG_BLOCK];
alignas(64) static bfloat16 log_f_block[3][LOG_BLOCK];
alignas(64) static bfloat16 log_e_block[LOG_BLOCK];

// The passes over a block. Restrict parameters, not locals, are what tell LLVM
// a pass's stores cannot reach its next loads, so each loop pipelines.

// x + offset = 2^e m: e as bf16, and f = m - 1 as three limbs.
static inline __attribute__((always_inline)) void
log_prep(const float *__restrict x, float offset, bfloat16 *__restrict ef,
         bfloat16 *__restrict fh, bfloat16 *__restrict fm,
         bfloat16 *__restrict fl, bf16_vec one) {
  for (int i = 0; i < LOG_BLOCK; i += f32_lanes) {
    auto bits = aie::add(f32_acc(aie::load_v<f32_lanes>(x + i)), offset)
                    .to_vector<float>()
                    .cast_to<int32_t>();
    auto e = aie::downshift(
        aie::sub(bits, aie::broadcast<int32_t, f32_lanes>(0x3f3504f3)), 23);
    auto m = aie::sub(bits, aie::upshift(e, 23)).cast_to<float>();
    auto e_magic = aie::add(e, aie::broadcast<int32_t, f32_lanes>(0x4b400000));
    aie::store_v(ef + i, aie::add(f32_acc(e_magic.cast_to<float>()), -12582912.0f)
                             .to_vector<bfloat16>());
    bf16_vec h, m_, l;
    split3(aie::msc(f32_acc(m), one, one), h, m_, l, one);
    aie::store_v(fh + i, h);
    aie::store_v(fm + i, m_);
    aie::store_v(fl + i, l);
  }
}

// The first step, from q = log_q[9], whose limbs are constant.
static inline __attribute__((always_inline)) void
log_first(const bfloat16 *__restrict fh, const bfloat16 *__restrict fm,
          const bfloat16 *__restrict fl, float *__restrict q_out, bf16_vec qh,
          bf16_vec qm, bf16_vec ql, float c) {
  for (int i = 0; i < LOG_BLOCK; i += f32_lanes) {
    const bf16_vec h = aie::load_v<f32_lanes>(fh + i);
    const bf16_vec m = aie::load_v<f32_lanes>(fm + i);
    const bf16_vec l = aie::load_v<f32_lanes>(fl + i);
    f32_acc q = aie::mul(ql, h);
    q = aie::mac(aie::mac(q, qm, m), qh, l);
    q = aie::mac(aie::mac(q, qm, h), qh, m);
    q = aie::add(aie::mac(q, qh, h), c);
    aie::store_v(q_out + i, q.to_vector<float>());
  }
}

// One Horner step, q_in to q_out.
static inline __attribute__((always_inline)) void
log_pass(const float *__restrict q_in, float *__restrict q_out,
         const bfloat16 *__restrict fh, const bfloat16 *__restrict fm,
         const bfloat16 *__restrict fl, float c, bf16_vec one) {
  for (int i = 0; i < LOG_BLOCK; i += f32_lanes) {
    const f32_acc q = log_step(f32_acc(aie::load_v<f32_lanes>(q_in + i)),
                               aie::load_v<f32_lanes>(fh + i),
                               aie::load_v<f32_lanes>(fm + i),
                               aie::load_v<f32_lanes>(fl + i), c, one);
    aie::store_v(q_out + i, q.to_vector<float>());
  }
}

// e ln 2 + f q, rounded once to bf16.
static inline __attribute__((always_inline)) void
log_last(const float *__restrict q_in, const bfloat16 *__restrict fh,
         const bfloat16 *__restrict fm, const bfloat16 *__restrict fl,
         const bfloat16 *__restrict ef, bfloat16 *__restrict y, bf16_vec one,
         bf16_vec ln2_hi, bf16_vec ln2_mid, bf16_vec ln2_lo) {
  for (int i = 0; i < LOG_BLOCK; i += f32_lanes) {
    const bf16_vec h = aie::load_v<f32_lanes>(fh + i);
    const bf16_vec m = aie::load_v<f32_lanes>(fm + i);
    const bf16_vec l = aie::load_v<f32_lanes>(fl + i);
    const bf16_vec e = aie::load_v<f32_lanes>(ef + i);
    const f32_acc q = f32_acc(aie::load_v<f32_lanes>(q_in + i));
    bf16_vec qh, qm, ql;
    split3(q, qh, qm, ql, one);
    f32_acc r = aie::mul(e, ln2_lo);
    r = aie::mac(r, e, ln2_mid);
    r = aie::mac(aie::mac(aie::mac(r, ql, h), qm, m), qh, l);
    r = aie::mac(aie::mac(aie::mac(r, qm, h), qh, m), qh, h);
    aie::store_v(y + i, aie::mac(r, e, ln2_hi).to_vector<bfloat16>());
  }
}
#endif

// log(x + offset) of float32, rounded once to bf16, for x + offset a positive
// normal float32. Multiplies as f32_split.h does.
extern "C" void log_f32_bf16(float *restrict x, bfloat16 *restrict y, int32_t n,
                             float offset) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec one = bf16_splat(0x3f80);
  // ln 2 as three bf16 limbs.
  const bf16_vec ln2_hi = bf16_splat(0x3f31);
  const bf16_vec ln2_mid = bf16_splat(0x3ae4);
  const bf16_vec ln2_lo = bf16_splat(0x35c0);
#if LOG_STAGED
  // Each Horner step is its own pass over a block, q held in float32 between
  // passes: the short single loops pipeline where the nested one spilled.
  // Per element the operations and their order are the nested loop's.
  bf16_vec q9h, q9m, q9l;
  split3(aie::add(f32_acc(aie::broadcast<float, f32_lanes>(0.0f)), log_q[9]),
         q9h, q9m, q9l, one);
  for (int b = 0; b < LOG_ELEMS; b += LOG_BLOCK) {
    log_prep(x + b, offset, log_e_block, log_f_block[0], log_f_block[1],
             log_f_block[2], one);
    log_first(log_f_block[0], log_f_block[1], log_f_block[2], log_q_block[0],
              q9h, q9m, q9l, log_q[8]);
    for (int k = 7; k >= 0; k--)
      log_pass(log_q_block[(k + 1) & 1], log_q_block[k & 1], log_f_block[0],
               log_f_block[1], log_f_block[2], log_q[k], one);
    log_last(log_q_block[0], log_f_block[0], log_f_block[1], log_f_block[2],
             log_e_block, y + b, one, ln2_hi, ln2_mid, ln2_lo);
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < LOG_ELEMS; i += f32_lanes) {
    auto bits = aie::add(f32_acc(aie::load_v<f32_lanes>(x + i)), offset)
                    .to_vector<float>()
                    .cast_to<int32_t>();
    // z = 2^e m with m in [sqrt(1/2), sqrt(2)); downshift floors in any mode.
    auto e = aie::downshift(
        aie::sub(bits, aie::broadcast<int32_t, f32_lanes>(0x3f3504f3)), 23);
    auto m = aie::sub(bits, aie::upshift(e, 23)).cast_to<float>();
    // e as a float: 1.5 * 2^23 + e has e in its low mantissa bits.
    auto e_magic = aie::add(e, aie::broadcast<int32_t, f32_lanes>(0x4b400000));
    bf16_vec ef = aie::add(f32_acc(e_magic.cast_to<float>()), -12582912.0f)
                      .to_vector<bfloat16>();
    bf16_vec fh, fm, fl;
#if AIE_ARCH_AIE2
    // As in split3: m - 1 in halves, so m < 1 is not rounded to 1's exponent.
    const bf16_vec half = bf16_splat(0x3f00);
    f32_acc f = aie::msc(aie::msc(f32_acc(m), one, half), one, half);
#else
    f32_acc f = aie::msc(f32_acc(m), one, one);
#endif
    split3(f, fh, fm, fl, one);

    f32_acc q =
        aie::add(f32_acc(aie::broadcast<float, f32_lanes>(0.0f)), log_q[9]);
#if AIE_TUNED_AIE2
    AIE_LOOP_UNROLL_FULL
#endif
    for (int k = 8; k >= 0; k--)
      q = log_step(q, fh, fm, fl, log_q[k], one);
    bf16_vec qh, qm, ql;
    split3(q, qh, qm, ql, one);
    f32_acc r = aie::mul(ef, ln2_lo);
    r = aie::mac(r, ef, ln2_mid);
    r = aie::mac(aie::mac(aie::mac(r, ql, fh), qm, fm), qh, fl);
    r = aie::mac(aie::mac(aie::mac(r, qm, fh), qh, fm), qh, fh);
    aie::store_v(y + i, aie::mac(r, ef, ln2_hi).to_vector<bfloat16>());
  }
#endif
  aie::set_rounding(saved);
  event1();
}
