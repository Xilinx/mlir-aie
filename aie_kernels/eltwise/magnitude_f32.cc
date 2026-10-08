//===- magnitude_f32.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MAGNITUDE_ELEMS
#define MAGNITUDE_ELEMS n
#endif

// sqrt(re^2 + im^2) of complex float32 held as [re | im], n of each. Every
// product is of bf16 limbs, exact in the float32 accumulator.
extern "C" void magnitude_f32(float *restrict x, float *restrict out,
                              int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec one = bf16_splat(0x3f80);
  const bf16_vec half = bf16_splat(0x3f00);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < MAGNITUDE_ELEMS; i += f32_lanes) {
    bf16_vec rh, rm, rl, ih, im, il;
    split3(f32_acc(aie::load_v<f32_lanes>(x + i)), rh, rm, rl, one);
    split3(f32_acc(aie::load_v<f32_lanes>(x + MAGNITUDE_ELEMS + i)), ih, im, il,
           one);
    f32_acc p = aie::mul(rm, rm);
    p = aie::mac(p, im, im);
    p = aie::mac(aie::mac(p, rh, rl), rh, rl);
    p = aie::mac(aie::mac(p, ih, il), ih, il);
    p = aie::mac(aie::mac(p, rh, rm), rh, rm);
    p = aie::mac(aie::mac(p, ih, im), ih, im);
    p = aie::mac(aie::mac(p, rh, rh), ih, ih);

    // 1 / sqrt(p): the bit-trick estimate, 3.4e-2 off, then a Newton step to
    // about 4e-3. p = 0 gives a finite estimate, and s = 0.
    auto bits = p.to_vector<float>().cast_to<int32_t>();
    auto y0_bits = aie::sub(aie::broadcast<int32_t, f32_lanes>(0x5f3759df),
                            aie::downshift(bits, 1));
    bf16_vec y0 = f32_acc(y0_bits.cast_to<float>()).to_vector<bfloat16>();
    bf16_vec ph, pm, pl;
    split3(p, ph, pm, pl, one);
    bf16_vec py = aie::mac(aie::mul(pm, y0), ph, y0).to_vector<bfloat16>();
    f32_acc r =
        aie::msc(f32_acc(aie::broadcast<float, f32_lanes>(1.0f)), py, y0);
    bf16_vec y0_half = aie::mul(y0, half).to_vector<bfloat16>();
    bf16_vec y = aie::mac(aie::mul(y0, one), r.to_vector<bfloat16>(), y0_half)
                     .to_vector<bfloat16>();
    bf16_vec y_half = aie::mul(y, half).to_vector<bfloat16>();

    // s = p y, then s += y / 2 (p - s^2): each step multiplies the error by
    // y's, and p - s^2 is taken from s's exact limb products.
    f32_acc s = aie::mac(aie::mac(aie::mul(pl, y), pm, y), ph, y);
    for (int k = 0; k < 3; k++) {
      bf16_vec sh, sm, sl;
      split3(s, sh, sm, sl, one);
      f32_acc e = aie::msc(p, sh, sh);
      e = aie::msc(aie::msc(e, sh, sm), sh, sm);
      e = aie::msc(aie::msc(e, sh, sl), sh, sl);
      e = aie::msc(e, sm, sm);
      bf16_vec eh = e.to_vector<bfloat16>();
      bf16_vec em = aie::msc(e, eh, one).to_vector<bfloat16>();
      s = aie::mac(aie::mac(s, em, y_half), eh, y_half);
    }
    aie::store_v(out + i, s.to_vector<float>());
  }
  aie::set_rounding(saved);
  event1();
}
