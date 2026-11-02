// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_SOFT_CAP_F32_H
#define AIE_KERNELS_COMMON_SOFT_CAP_F32_H

#include "exp2_bf16.h"
#include "f32_split.h"

// cap * tanh(s / cap) * log2(e) over 32 lanes, to float32: k0 (2r - 1) of |s|
// with k0 = cap log2(e), r = 1 / (1 + 2^a) and a = -|s| * 2 log2(e) / cap
// <= 0, 2^a flushing to 0 saturating it at k0; then s's sign.
static inline __attribute__((always_inline)) aie::accum<accfloat, 32>
soft_cap_log2e(aie::accum<accfloat, 32> s, float cap) {
  constexpr unsigned N = 32;
  using Acc = aie::accum<accfloat, N>;
  auto splat = [](float c) { return Acc(aie::broadcast<float, N>(c)); };
  const float k0 = cap * 1.44269504088896341f;
  const Limbs<N> a_limbs = bf16_limbs(splat(2.88539008177792682f / cap));
  const Limbs<N> k1_limbs = bf16_limbs(splat(2.0f * k0));
  // The sign by the bits' compare and -s in the accumulator: Peano
  // cannot legalize a 32-lane f32 fneg or 1024-bit OR.
  const auto sv = s.to_vector<float>();
  const auto negative =
      aie::lt(sv.cast_to<int32_t>(), aie::zeros<int32_t, N>());
  const auto neg = aie::sub(aie::zeros<accfloat, N>(), sv);
  const Acc neg_abs(aie::select(neg.to_vector<float>(), sv, negative));
  const aie::vector<float, N> a =
      mac_limbs(aie::zeros<accfloat, N>(), bf16_limbs(neg_abs), a_limbs)
          .to_vector<float>();
  const Exp2F32 e2a = exp2_f32_32(a);
  const Acc d =
      aie::add(splat(1.0f), aie::select(e2a.bits.cast_to<float>(),
                                        aie::zeros<float, N>(), e2a.below));

  // r = 1 / d, d in [1, 2]: a quadratic within 2%, then two Newton steps
  // r -= r (d r - 1), the first from r in bf16.
  const Limbs<N> dl = bf16_limbs(d);
  const auto q = aie::mac(splat(-1.4515766f), dl.hi, bf16_splat<N>(0x3ea5))
                     .to_vector<bfloat16>();
  const auto r0 = aie::mac(splat(2.1190739f), q, dl.hi).to_vector<bfloat16>();
  Limbs<N> e = bf16_limbs(aie::mac(
      aie::mac(aie::mac(splat(-1.0f), dl.hi, r0), dl.mid, r0), dl.lo, r0));
  const Acc r1 = aie::msc(aie::msc(Acc(r0), r0, e.hi), r0, e.mid);
  const Limbs<N> rl = bf16_limbs(r1);
  e = bf16_limbs(mac_limbs(splat(-1.0f), dl, rl));
  const Acc r =
      aie::msc(aie::msc(aie::msc(r1, rl.hi, e.hi), rl.hi, e.mid), rl.mid, e.hi);

  const auto t = mac_limbs(splat(-k0), bf16_limbs(r), k1_limbs);
  return Acc(
      aie::select(t.to_vector<float>(),
                  aie::sub(aie::zeros<accfloat, N>(), t.to_vector<float>())
                      .to_vector<float>(),
                  negative));
}

#endif
