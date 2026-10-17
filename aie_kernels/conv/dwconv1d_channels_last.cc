//===- dwconv1d_channels_last.cc --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h" // AIE_LOOP_UNROLL_FULL

#include <aie_api/aie.hpp>

// Depthwise conv1d over a channels-last layout, with an optional bias and an
// optional clamp:
//   y[c] = clamp(sum_{t=0..K-1} x_t[c] * w_t[c] + b[c], lo, hi),  c in [0, C)
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
/// from one base as `w + t * stride` miscompiled under the full unroll below
/// before llvm-aie#1313's fix: in the first 32-lane group, planes 0..K-2 all
/// resolved to plane 0.
// The taps go outermost over four groups at a time: each group still sums its
// taps in order, but ten pointers walked one group at a time leave the loads
// single-issued between pointer moves. AIE2 groups are 32 lanes; AIE2P's
// vmac.f fills a whole accumulator register at 64.
template <int K, int C, bool CLAMP, bool BIAS, int vec_size>
static inline void dwconv1d_channels_last_grouped(
    const bf16 *const *__restrict w, const bf16 *const *__restrict x,
    const bf16 *__restrict b, bf16 lo, bf16 hi, bf16 *__restrict y) {
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
    if constexpr (BIAS) {
      AIE_LOOP_UNROLL_FULL
      for (int g = 0; g < G; g++) {
        if (g0 + g < NG) {
          const int o = (g0 + g) * vec_size;
          acc[g] = aie::mac(acc[g], aie::load_v<vec_size>(b + o),
                            aie::broadcast<bf16, vec_size>(1.0f));
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

template <int K, int C, bool CLAMP, bool BIAS>
static inline void dwconv1d_channels_last_generic(
    const bf16 *const *__restrict w, const bf16 *const *__restrict x,
    const bf16 *__restrict b, bf16 lo, bf16 hi, bf16 *__restrict y) {
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
    if constexpr (BIAS) {
      acc = aie::mac(acc, aie::load_v<vec_size>(b + o),
                     aie::broadcast<bf16, vec_size>(1.0f));
    }

    aie::vector<bf16, vec_size> y_vec = acc.template to_vector<bf16>();
    if constexpr (CLAMP) {
      y_vec = aie::clamp(y_vec, lo, hi);
    }
    aie::store_v(y + o, y_vec);
  }
}

template <int K, int C, bool CLAMP, bool BIAS, bool GROUPED>
static inline void dwconv1d_channels_last_body(const bf16 *const *__restrict w,
                                               const bf16 *const *__restrict x,
                                               const bf16 *__restrict b,
                                               bf16 lo, bf16 hi,
                                               bf16 *__restrict y) {
  if constexpr (GROUPED)
    dwconv1d_channels_last_grouped<K, C, CLAMP, BIAS, 64>(w, x, b, lo, hi, y);
  else
    dwconv1d_channels_last_generic<K, C, CLAMP, BIAS>(w, x, b, lo, hi, y);
}

// Walks C in chunks of CH channels in a loop kept rolled. Unrolled over all of
// C, the scheduler hoisted every chunk's loads and parked them on the stack,
// which grew by about 8 B per channel.
template <int K, int C, bool CLAMP, bool BIAS, int CH, bool GROUPED>
static inline void dwconv1d_channels_last_chunked(
    const bf16 *const *__restrict w, const bf16 *const *__restrict x,
    const bf16 *__restrict b, bf16 lo, bf16 hi, bf16 *__restrict y) {
  constexpr int TAIL = C % CH;
  const bf16 *wc[K];
  const bf16 *xc[K];
  AIE_LOOP_NO_UNROLL
  for (int o = 0; o < C - TAIL; o += CH) {
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < K; t++) {
      wc[t] = w[t] + o;
      xc[t] = x[t] + o;
    }
    dwconv1d_channels_last_body<K, CH, CLAMP, BIAS, GROUPED>(
        wc, xc, BIAS ? b + o : b, lo, hi, y + o);
  }
  if constexpr (TAIL != 0) {
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < K; t++) {
      wc[t] = w[t] + (C - TAIL);
      xc[t] = x[t] + (C - TAIL);
    }
    dwconv1d_channels_last_body<K, TAIL, CLAMP, BIAS, GROUPED>(
        wc, xc, BIAS ? b + (C - TAIL) : b, lo, hi, y + (C - TAIL));
  }
}

template <int K, int C, bool CLAMP, bool BIAS>
static inline void dwconv1d_channels_last_impl(const bf16 *const *__restrict w,
                                               const bf16 *const *__restrict x,
                                               const bf16 *__restrict b,
                                               bf16 lo, bf16 hi,
                                               bf16 *__restrict y) {
  // Up to 256 channels the generic loop is fastest unrolled whole; past that
  // it takes four groups at a time, since a 256-channel chunk plus its tail
  // parked as much as the whole loop did.
  constexpr int GENERIC_CHUNK = C <= 256 ? 256 : 128;
#if AIE_TUNED_AIE2
  dwconv1d_channels_last_grouped<K, C, CLAMP, BIAS, 32>(w, x, b, lo, hi, y);
#elif AIE_TUNED_AIE2P
  // Four 32-lane groups measured slower than the generic loop at C = 96. A
  // chunk of four 64-lane groups keeps C = 256 at its unrolled speed; smaller
  // chunks measured 1.6x to 3x slower.
  if constexpr (C % 64 == 0)
    dwconv1d_channels_last_chunked<K, C, CLAMP, BIAS, 256, true>(w, x, b, lo,
                                                                 hi, y);
  else
    dwconv1d_channels_last_chunked<K, C, CLAMP, BIAS, GENERIC_CHUNK, false>(
        w, x, b, lo, hi, y);
#else
  dwconv1d_channels_last_chunked<K, C, CLAMP, BIAS, GENERIC_CHUNK, false>(
      w, x, b, lo, hi, y);
#endif
}

#ifndef DWCONV1D_CL_C
#define DWCONV1D_CL_C 256
#endif
#ifndef DWCONV1D_CL_CLAMP
#define DWCONV1D_CL_CLAMP 1
#endif
#ifndef DWCONV1D_CL_TAPS
#define DWCONV1D_CL_TAPS 5
#endif
#ifndef DWCONV1D_CL_BIAS
#define DWCONV1D_CL_BIAS 0
#endif

// One entry point a build, named for its taps and its bias: K sets the
// argument count, since both operands are separate ObjectFifo elements, so a
// different K is a different signature.
#if DWCONV1D_CL_TAPS == 2
#define DWCONV1D_CL_PLANES(p) bf16 *__restrict p##_0, bf16 *__restrict p##_1
#define DWCONV1D_CL_LIST(p) p##_0, p##_1
#elif DWCONV1D_CL_TAPS == 3
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2
#define DWCONV1D_CL_LIST(p) p##_0, p##_1, p##_2
#elif DWCONV1D_CL_TAPS == 4
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2,      \
      bf16 *__restrict p##_3
#define DWCONV1D_CL_LIST(p) p##_0, p##_1, p##_2, p##_3
#elif DWCONV1D_CL_TAPS == 5
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2,      \
      bf16 *__restrict p##_3, bf16 *__restrict p##_4
#define DWCONV1D_CL_LIST(p) p##_0, p##_1, p##_2, p##_3, p##_4
#elif DWCONV1D_CL_TAPS == 6
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2,      \
      bf16 *__restrict p##_3, bf16 *__restrict p##_4, bf16 *__restrict p##_5
#define DWCONV1D_CL_LIST(p) p##_0, p##_1, p##_2, p##_3, p##_4, p##_5
#elif DWCONV1D_CL_TAPS == 7
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2,      \
      bf16 *__restrict p##_3, bf16 *__restrict p##_4, bf16 *__restrict p##_5,  \
      bf16 *__restrict p##_6
#define DWCONV1D_CL_LIST(p) p##_0, p##_1, p##_2, p##_3, p##_4, p##_5, p##_6
#elif DWCONV1D_CL_TAPS == 8
#define DWCONV1D_CL_PLANES(p)                                                  \
  bf16 *__restrict p##_0, bf16 *__restrict p##_1, bf16 *__restrict p##_2,      \
      bf16 *__restrict p##_3, bf16 *__restrict p##_4, bf16 *__restrict p##_5,  \
      bf16 *__restrict p##_6, bf16 *__restrict p##_7
#define DWCONV1D_CL_LIST(p)                                                    \
  p##_0, p##_1, p##_2, p##_3, p##_4, p##_5, p##_6, p##_7
#else
#error "DWCONV1D_CL_TAPS must be 2 to 8"
#endif

#define DWCONV1D_CL_CAT_(a, b, c) a##b##c
#define DWCONV1D_CL_CAT(a, b, c) DWCONV1D_CL_CAT_(a, b, c)
#if DWCONV1D_CL_BIAS
#define DWCONV1D_CL_NAME                                                       \
  DWCONV1D_CL_CAT(dwconv1d_channels_last_k, DWCONV1D_CL_TAPS, _bias_bf16)
#define DWCONV1D_CL_BIAS_ARG bf16 *__restrict b,
#else
#define DWCONV1D_CL_NAME                                                       \
  DWCONV1D_CL_CAT(dwconv1d_channels_last_k, DWCONV1D_CL_TAPS, _bf16)
#define DWCONV1D_CL_BIAS_ARG
#endif

extern "C" {

// The weight planes come in as K pointers like the taps do, rather than one
// base this indexes by row and stride. That keeps the plane layout the
// design's business -- the planes need not be one buffer, or evenly spaced --
// and it is the form the taps already proved correct under the full unroll.
// The bias, where the build has one, is one more plane.
//
// lo/hi arrive as float buffers because they are runtime parameters the host
// writes.
void DWCONV1D_CL_NAME(DWCONV1D_CL_PLANES(w), DWCONV1D_CL_PLANES(x),
                      DWCONV1D_CL_BIAS_ARG bf16 *__restrict y, float *lo_buffer,
                      float *hi_buffer) {
  event0();
  constexpr int C = DWCONV1D_CL_C;
  constexpr int K = DWCONV1D_CL_TAPS;

  const bf16 *const wp[K] = {DWCONV1D_CL_LIST(w)};
  const bf16 *const xp[K] = {DWCONV1D_CL_LIST(x)};
#if DWCONV1D_CL_BIAS
  const bf16 *const bp = b;
#else
  const bf16 *const bp = nullptr;
#endif

  dwconv1d_channels_last_impl<K, C, (bool)DWCONV1D_CL_CLAMP,
                              (bool)DWCONV1D_CL_BIAS>(
      wp, xp, bp, (bf16)lo_buffer[0], (bf16)hi_buffer[0], y);
  event1();
}

} // extern "C"
