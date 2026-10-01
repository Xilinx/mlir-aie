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

#include "../common/scalar_f32.h" // scalar_mul
#include "utils.h"

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
  uint32_t i_u32 = *(uint32_t *)&divrms;
  i_u32 = 0x5f3759df - (i_u32 >> 1);
  divrms = *(float *)&i_u32;
  divrms = scalar_mul(
      divrms, (threehalfs - scalar_mul(scalar_mul(x2, divrms), divrms)));
  divrms = scalar_mul(
      divrms, (threehalfs - scalar_mul(scalar_mul(x2, divrms), divrms)));
  return divrms;
}

/// \brief y = x * rsqrt(mean(x^2) + eps) * w, over D elements.
template <int D>
void rms_norm(bf16 *y, const bf16 *x, const bf16 *w) {
  constexpr int vector_size = 16;
  const float divrms = rms_scale<D>(x);
  for (int i = 0; i < D / vector_size; i++) {
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(x);
    aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(w);
    aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
    aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
    aie::store_v(y, o_vec);
    x += vector_size;
    y += vector_size;
    w += vector_size;
  }
}

/// \brief y = x * rsqrt(mean(x^2) + eps), over D elements.
template <int D>
void rms_norm_unweighted(bf16 *y, const bf16 *x) {
  constexpr int vector_size = 16;
  const float divrms = rms_scale<D>(x);
  for (int i = 0; i < D / vector_size; i++) {
    aie::accum<accfloat, vector_size> x_acc;
    x_acc.from_vector(aie::load_v<vector_size>(x));
    aie::vector<float, vector_size> x_float = x_acc.template to_vector<float>();
    aie::vector<bf16, vector_size> o_vec = aie::mul(x_float, divrms);
    aie::store_v(y, o_vec);
    x += vector_size;
    y += vector_size;
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
