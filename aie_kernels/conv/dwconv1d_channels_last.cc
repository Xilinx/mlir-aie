//===- dwconv1d_channels_last.cc --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h" // AIE_LOOP_UNROLL_FULL

#include <aie_api/aie.hpp>

// Depthwise conv1d over a channels-last layout, with an optional clamp:
//   y[c] = clamp(sum_{t=0..K-1} x_t[c] * w_t[c], lo, hi),  c in [0, C)
//
// One call emits one output timestep for all C channels. The K taps arrive as
// K separate base pointers, oldest first, so a depth-K ObjectFifo is itself the
// rotation and none is hand-rolled here.
//
// Counterpart to dwconv1d_channels_first.cc. Layout picks the vectorization
// axis, so neither subsumes the other; see PICKING ONE below.

using bf16 = bfloat16;

// Denser per instruction than the channels-first form, but only worth it when
// the data is already in this layout; the README's "Choosing a depthwise
// conv1d" section has the measurement and the tradeoffs.

/// K taps over C channels, accumulating in float and narrowing on store.
///
/// Both operands arrive as K independent pointers. Deriving the weight planes
/// from one base as `w + t * stride` miscompiled under the full unroll below:
/// in the first 32-lane group, planes 0..K-2 all resolved to plane 0.
// The taps go outermost over four groups at a time: each group still sums its
// taps in order, but ten pointers walked one group at a time leave the loads
// single-issued between pointer moves. AIE2 groups are 32 lanes; AIE2P's
// vmac.f fills a whole accumulator register at 64.
template <int K, int C, bool CLAMP, int vec_size>
static inline void
dwconv1d_channels_last_grouped(const bf16 *const *__restrict w,
                               const bf16 *const *__restrict x, bf16 lo,
                               bf16 hi, bf16 *__restrict y) {
  static_assert(C % vec_size == 0, "C must be a multiple of the group width");
  constexpr int G = 4;
  constexpr int NG = C / vec_size;

  AIE_LOOP_UNROLL_FULL
  for (int g0 = 0; g0 < NG; g0 += G) {
    aie::accum<accfloat, vec_size> acc[G];
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < G; g++) {
      if (g0 + g < NG) {
        const int o = (g0 + g) * vec_size;
        acc[g] = aie::mul(aie::load_v<vec_size>(x[0] + o),
                          aie::load_v<vec_size>(w[0] + o));
      }
    }
    AIE_LOOP_UNROLL_FULL
    for (int t = 1; t < K; t++) {
      AIE_LOOP_UNROLL_FULL
      for (int g = 0; g < G; g++) {
        if (g0 + g < NG) {
          const int o = (g0 + g) * vec_size;
          acc[g] = aie::mac(acc[g], aie::load_v<vec_size>(x[t] + o),
                            aie::load_v<vec_size>(w[t] + o));
        }
      }
    }
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < G; g++) {
      if (g0 + g < NG) {
        const int o = (g0 + g) * vec_size;
        aie::vector<bf16, vec_size> y_vec = acc[g].template to_vector<bf16>();
        if constexpr (CLAMP) {
          y_vec = aie::clamp(y_vec, lo, hi);
        }
        aie::store_v(y + o, y_vec);
      }
    }
  }
}

template <int K, int C, bool CLAMP>
static inline void
dwconv1d_channels_last_generic(const bf16 *const *__restrict w,
                               const bf16 *const *__restrict x, bf16 lo,
                               bf16 hi, bf16 *__restrict y) {
  constexpr int vec_size = 32;
  static_assert(C % vec_size == 0, "C must be a multiple of the 32-lane store");

  // Full unroll rather than a trip-count hint: under Peano AIE_LOOP_RANGE is a
  // hint only, and has been observed to leave a short MAC loop miscompiled to a
  // single iteration.
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < C / vec_size; i++) {
    const int o = i * vec_size;

    aie::accum<accfloat, vec_size> acc = aie::mul(
        aie::load_v<vec_size>(x[0] + o), aie::load_v<vec_size>(w[0] + o));

    AIE_LOOP_UNROLL_FULL
    for (int t = 1; t < K; t++) {
      acc = aie::mac(acc, aie::load_v<vec_size>(x[t] + o),
                     aie::load_v<vec_size>(w[t] + o));
    }

    aie::vector<bf16, vec_size> y_vec = acc.template to_vector<bf16>();
    if constexpr (CLAMP) {
      y_vec = aie::clamp(y_vec, lo, hi);
    }
    aie::store_v(y + o, y_vec);
  }
}

template <int K, int C, bool CLAMP>
static inline void dwconv1d_channels_last_impl(const bf16 *const *__restrict w,
                                               const bf16 *const *__restrict x,
                                               bf16 lo, bf16 hi,
                                               bf16 *__restrict y) {
#if AIE_TUNED_AIE2
  dwconv1d_channels_last_grouped<K, C, CLAMP, 32>(w, x, lo, hi, y);
#elif AIE_TUNED_AIE2P
  // Four 32-lane groups measured slower than the generic loop at C = 96.
  if constexpr (C % 64 == 0)
    dwconv1d_channels_last_grouped<K, C, CLAMP, 64>(w, x, lo, hi, y);
  else
    dwconv1d_channels_last_generic<K, C, CLAMP>(w, x, lo, hi, y);
#else
  dwconv1d_channels_last_generic<K, C, CLAMP>(w, x, lo, hi, y);
#endif
}

#ifndef DWCONV1D_CL_C
#define DWCONV1D_CL_C 256
#endif
#ifndef DWCONV1D_CL_CLAMP
#define DWCONV1D_CL_CLAMP 1
#endif

extern "C" {

// 5-tap entry point. K is fixed here rather than templated because it sets the
// argument count: both operands are separate ObjectFifo elements, so a
// different K is a different signature and wants its own entry point.
//
// The weight planes come in as five pointers like the taps do, rather than one
// base this indexes by row and stride. That keeps the plane layout the
// design's business -- the planes need not be one buffer, or evenly spaced --
// and it is the form the taps already proved correct under the full unroll.
//
// lo/hi arrive as float buffers because they are runtime parameters the host
// writes.
void dwconv1d_channels_last_k5_bf16(bf16 *__restrict w_0, bf16 *__restrict w_1,
                                    bf16 *__restrict w_2, bf16 *__restrict w_3,
                                    bf16 *__restrict w_4, bf16 *__restrict x_0,
                                    bf16 *__restrict x_1, bf16 *__restrict x_2,
                                    bf16 *__restrict x_3, bf16 *__restrict x_4,
                                    bf16 *__restrict y, float *lo_buffer,
                                    float *hi_buffer) {
  event0();
  constexpr int C = DWCONV1D_CL_C;

  const bf16 *const wp[5] = {w_0, w_1, w_2, w_3, w_4};
  const bf16 *const xp[5] = {x_0, x_1, x_2, x_3, x_4};

  dwconv1d_channels_last_impl<5, C, (bool)DWCONV1D_CL_CLAMP>(
      wp, xp, (bf16)lo_buffer[0], (bf16)hi_buffer[0], y);
  event1();
}

} // extern "C"
