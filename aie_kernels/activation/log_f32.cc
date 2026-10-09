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

#ifndef LOG_ELEMS
#define LOG_ELEMS n
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
    for (int k = 8; k >= 0; k--) {
      bf16_vec qh, qm, ql;
      split3(q, qh, qm, ql, one);
      q = aie::mul(ql, fh);
      q = aie::mac(aie::mac(q, qm, fm), qh, fl);
      q = aie::mac(aie::mac(q, qm, fh), qh, fm);
      q = aie::add(aie::mac(q, qh, fh), log_q[k]);
    }
    bf16_vec qh, qm, ql;
    split3(q, qh, qm, ql, one);
    f32_acc r = aie::mul(ef, ln2_lo);
    r = aie::mac(r, ef, ln2_mid);
    r = aie::mac(aie::mac(aie::mac(r, ql, fh), qm, fm), qh, fl);
    r = aie::mac(aie::mac(aie::mac(r, qm, fh), qh, fm), qh, fh);
    aie::store_v(y + i, aie::mac(r, ef, ln2_hi).to_vector<bfloat16>());
  }
  aie::set_rounding(saved);
  event1();
}
