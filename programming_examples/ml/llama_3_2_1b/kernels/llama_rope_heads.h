//===- llama_rope_heads.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RoPE over kNHeads int8 Q heads, [body int8 kHD x kNHeads | scale tail 8 B
// per head], the tail passed through.
//===----------------------------------------------------------------------===//

#ifndef LLAMA_ROPE_HEADS_H
#define LLAMA_ROPE_HEADS_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

template <int kHD, int kNHeads>
static inline void llama_rope_heads(const int8_t *restrict x,
                                    const bfloat16 *restrict cs_packed,
                                    int8_t *restrict out) {
  constexpr int kHalf = kHD / 2;
  constexpr int kBody = kHD * kNHeads;
  constexpr int kTail = kNHeads * 8;
  static_assert(kHD % 2 == 0, "head_dim must be even");
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);

  const bfloat16 *cos = cs_packed;
  const bfloat16 *sin = cs_packed + kHD;

  // cos/sin are the same for every head, so widen them to f32 once outside
  // the head loop (exact bf16->f32, see llama_rmsnorm_int8.cc for why that
  // holds). Products accumulate in accfloat and round to f32 once, same
  // discipline as aie_kernels/datamovement/rope.cc's rope_halves_step.
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

  for (int h = 0; h < kNHeads; h++) {
    int base = h * kHD;
    ::aie::vector<int8, kHalf> x1v = ::aie::load_v<kHalf>(x + base);
    ::aie::vector<int8, kHalf> x2v = ::aie::load_v<kHalf>(x + base + kHalf);
    ::aie::vector<float, kHalf> x1f = ::aie::to_float<float>(x1v);
    ::aie::vector<float, kHalf> x2f = ::aie::to_float<float>(x2v);

    ::aie::vector<float, kHalf> o1 =
        ::aie::msc(::aie::mul(x1f, c1f), x2f, s1f).template to_vector<float>();
    ::aie::vector<float, kHalf> o2 =
        ::aie::mac(::aie::mul(x2f, c2f), x1f, s2f).template to_vector<float>();

    ::aie::store_v(out + base, vec_round_to_i8(o1));
    ::aie::store_v(out + base + kHalf, vec_round_to_i8(o2));
  }

  // Passthrough the per-head scale tail unchanged.
  copy_words<kTail>(out + kBody, x + kBody);
}

#endif // LLAMA_ROPE_HEADS_H
