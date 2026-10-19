//===- rms_norm.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RMSNorm and the residual add. Independent of the model geometry.
//
// The pointers must not be `restrict`: callers pass the same buffer as y and x.
#ifndef AIE_KERNELS_FLM_GEMMA4_RMS_NORM_H
#define AIE_KERNELS_FLM_GEMMA4_RMS_NORM_H

#include "../common/f32_split.h"
#include "../common/scalar_f32.h" // scalar_mul
#include "utils.h"

/// \brief bf16(x * w * s), bit for bit as
/// `aie::mul(aie::mul(x, w).to_vector<float>(), s)` gives it.
///
/// That f32 multiply is emulated: both operands split into three bf16 limbs
/// and nine limb products summed smallest first. x * w is the product of two
/// bf16, exact in two limbs, so its third limb is zero and its three products
/// drop out; the rest run in the emulation's order. The sum starts from +0, as
/// the emulation's zero products leave it.
static inline __attribute__((always_inline)) bf16_vec
weighted_norm(bf16_vec x, bf16_vec w, const Limbs<f32_lanes> &s) {
  const f32_acc xw = aie::mul(x, w);
  const bf16_vec hi = xw.to_vector<bfloat16>();
  const bf16_vec mid =
      aie::msc(xw, hi, aie::broadcast<bfloat16, f32_lanes>(1.0f))
          .to_vector<bfloat16>();
  f32_acc acc = aie::zeros<accfloat, f32_lanes>();
  acc = aie::mac(acc, mid, s.lo);
  acc = aie::mac(acc, hi, s.lo);
  acc = aie::mac(acc, mid, s.mid);
  acc = aie::mac(acc, mid, s.hi);
  acc = aie::mac(acc, hi, s.mid);
  return aie::mac(acc, hi, s.hi).to_vector<bfloat16>();
}

/// \brief rsqrt(mean(x^2) + eps) over D elements.
///
/// The reciprocal square root is the fast inverse square root with two Newton
/// steps, because it is the numeric contract with the FastFlowLM baseline.
/// Scalar float multiplies use scalar_mul, not `*`: AIE2P has no scalar fp32
/// multiply, and a plain `*` becomes a __mulsf3 call that blocks
/// vectorization. The sum of squares takes 16 lanes; a wider accumulator
/// reassociates it and changes the result.
template <int D>
float rms_scale(const bf16 *x) {
  const float epsilon = 1e-6;
  constexpr int vector_size = 16;
  constexpr float one_over_D = 1.0f / (float)D;

  aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
  for (int i = 0; i < D / vector_size; i++) {
    sum_squares = aie::mac_square(sum_squares, aie::load_v<vector_size>(x));
    x += vector_size;
  }

  float sum = aie::reduce_add(sum_squares.template to_vector<float>());
  sum = scalar_mul(sum, one_over_D);
  sum = sum + epsilon;
  const float threehalfs = 1.5F;
  float x2 = scalar_mul(sum, 0.5F);
  float divrms = sum;
  uint32_t i_u32 = __builtin_bit_cast(uint32_t, divrms);
  i_u32 = 0x5f3759df - (i_u32 >> 1);
  divrms = __builtin_bit_cast(float, i_u32);
  divrms = scalar_mul(
      divrms, (threehalfs - scalar_mul(scalar_mul(x2, divrms), divrms)));
  divrms = scalar_mul(
      divrms, (threehalfs - scalar_mul(scalar_mul(x2, divrms), divrms)));
  return divrms;
}

/// \brief y = x * rsqrt(mean(x^2) + eps) * w, over D elements.
template <int D>
void rms_norm(bf16 *y, const bf16 *x, const bf16 *w) {
  static_assert(D % (2 * f32_lanes) == 0);
  const Limbs<f32_lanes> s =
      bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(rms_scale<D>(x))));
  // Two vectors a trip, both loaded before either store: their limb chains
  // overlap, and y may still be x.
  for (int i = 0; i < D / (2 * f32_lanes); i++) {
    const bf16_vec x0 = aie::load_v<f32_lanes>(x);
    const bf16_vec x1 = aie::load_v<f32_lanes>(x + f32_lanes);
    const bf16_vec w0 = aie::load_v<f32_lanes>(w);
    const bf16_vec w1 = aie::load_v<f32_lanes>(w + f32_lanes);
    aie::store_v(y, weighted_norm(x0, w0, s));
    aie::store_v(y + f32_lanes, weighted_norm(x1, w1, s));
    x += 2 * f32_lanes;
    y += 2 * f32_lanes;
    w += 2 * f32_lanes;
  }
}

/// \brief y = x * rsqrt(mean(x^2) + eps), over D elements.
///
/// x is bf16, so its middle limb is zero too: three of `weighted_norm`'s
/// products remain, in the same order.
template <int D>
void rms_norm_unweighted(bf16 *y, const bf16 *x) {
  static_assert(D % f32_lanes == 0);
  const Limbs<f32_lanes> s =
      bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(rms_scale<D>(x))));
  for (int i = 0; i < D / f32_lanes; i++) {
    const bf16_vec v = aie::load_v<f32_lanes>(x);
    f32_acc acc = aie::zeros<accfloat, f32_lanes>();
    acc = aie::mac(acc, v, s.lo);
    acc = aie::mac(acc, v, s.mid);
    aie::store_v(y, aie::mac(acc, v, s.hi).to_vector<bfloat16>());
    x += f32_lanes;
    y += f32_lanes;
  }
}

/// \brief y = x + x_buf, and x_buf = y, over D elements.
///
/// x_buf becomes the running residual for the next block. y and x may alias.
template <int D>
void residual_add(bf16 *y, const bf16 *x_buf, const bf16 *x) {
  const bf16 *it_x = x;
  bf16 *it_y = y;
  bf16 *it_x_buf = const_cast<bf16 *>(x_buf);

  for (int i = 0; i < D / 16; i++) {
    aie::vector<bf16, 16> x_vec = aie::load_v<16>(it_x);
    aie::vector<bf16, 16> x_buf_vec = aie::load_v<16>(it_x_buf);
    auto out_vec = aie::add(x_vec, x_buf_vec);
    aie::store_v(it_y, out_vec);
    aie::store_v(it_x_buf, out_vec);
    it_x += 16;
    it_y += 16;
    it_x_buf += 16;
  }
}

#endif // AIE_KERNELS_FLM_GEMMA4_RMS_NORM_H
