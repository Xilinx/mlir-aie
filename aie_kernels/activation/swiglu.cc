//===- swiglu.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/tanh_f32.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SWIGLU_ELEMS
#define SWIGLU_ELEMS 1024
#endif

// (x w1) silu(x w2) of bf16, a silu(g) for a = x w1 and g = x w2 each rounded
// to bf16: sigmoid_f32 of g with a g for its factor, exact in f32, rounded
// once to bf16. From |a| = 2^64 a is taken down by 2^64 first and the 2^64
// given back through sigmoid_f32's exponent, so a g cannot overflow. Above
// g = 128, where silu is g, for NaN g and for a non-finite a, the output is
// a g. It is 0 where g is 0 or at most -712, as swiglu_ref's float64 silu is.
extern "C" void swiglu_bf16(bfloat16 *restrict x, bfloat16 *restrict w1,
                            bfloat16 *restrict w2, bfloat16 *restrict y) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec top = aie::broadcast<bfloat16, f32_lanes>(184.0f);
  const bf16_vec bottom = aie::broadcast<bfloat16, f32_lanes>(-184.0f);
  const bf16_vec down = aie::broadcast<bfloat16, f32_lanes>(0x1p-64f);
  const bf16_vec zero = aie::zeros<bfloat16, f32_lanes>();
  const auto u16 = [](uint16_t c) {
    return aie::broadcast<uint16_t, f32_lanes>(c);
  };
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < SWIGLU_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    const bf16_vec a =
        aie::mul(xv, aie::load_v<f32_lanes>(w1 + i)).to_vector<bfloat16>();
    const bf16_vec g =
        aie::mul(xv, aie::load_v<f32_lanes>(w2 + i)).to_vector<bfloat16>();
    const auto a_abs = aie::bit_and(a.cast_to<uint16_t>(), u16(0x7fff));
    const auto g_bits = g.cast_to<uint16_t>();
    const auto g_abs = aie::bit_and(g_bits, u16(0x7fff));
    const auto big = aie::ge(a_abs, u16(0x5f80));
    const bf16_vec a_held =
        aie::select(a, aie::mul(a, down).to_vector<bfloat16>(), big);
    f32_acc ga;
    ga.from_vector(g);
    const bf16_vec out =
        sigmoid_f32(ga,
                    bf16_limbs(f32_acc(
                        aie::mul(a_held, aie::max(aie::min(g, top), bottom)))),
                    aie::select(aie::zeros<int32_t, f32_lanes>(),
                                aie::broadcast<int32_t, f32_lanes>(64), big))
            .to_vector<bfloat16>();
    const auto pass = aie::gt(g.cast_to<int16_t>(),
                              aie::broadcast<int16_t, f32_lanes>(0x4300)) |
                      aie::gt(g_abs, u16(0x7f80)) | aie::ge(a_abs, u16(0x7f80));
    const auto zero_out =
        aie::eq(g_abs, u16(0)) |
        (aie::ge(g_bits, u16(0xc432)) & aie::le(g_bits, u16(0xff80)));
    aie::store_v(
        y + i, aie::select(
                   aie::select(out, aie::mul(a, g).to_vector<bfloat16>(), pass),
                   zero, zero_out));
  }
  aie::set_rounding(saved);
  event1();
}
