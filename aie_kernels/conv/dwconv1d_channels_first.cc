//===- dwconv1d_channels_first.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Depthwise conv1d over a channels-first tensor: one channel per call, time
// contiguous, 'same' padding, stride 1, bf16. Cross-correlation with no kernel
// flip, matching torch.nn.Conv1d. Taps are K scalars and the vectorization runs
// along time, via sliding_mul.
//
// This is the general one: runtime T, 'same' padding, optional bias. See
// dwconv1d_channels_last.cc for the transposed layout, and the README's
// "Choosing a depthwise conv1d" for which to reach for.
//
// The caller supplies the padded row [P zeros | T samples | P zeros | slack],
// P = (K-1)/2, with a fixed 16 elements of slack whatever K is so the aligned
// 16-wide loads never read past the buffer. in_pad must be 256-bit aligned and
// T a multiple of 16; dwconv1d.py's `_pad_input` builds one.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#if AIE_TUNED_AIE2
template <int K>
__attribute__((always_inline)) static inline ::aie::vector<bfloat16, 32>
shifted(const ::aie::vector<bfloat16, 32> &w0,
        const ::aie::vector<bfloat16, 32> &w1, int p) {
  if (p >= K)
    return ::aie::zeros<bfloat16, 32>();
  if (p == 0)
    return w0;
  return ::aie::shuffle_down_fill(w0, w1, p);
}

// AIE2 has no bf16 sliding multiply: a bf16 vmac.f sums, per lane i of 16,
// a[i] * b[i] and a[16 + i] * b[16 + i]. One shift of a 48-sample window
// serves a tap for two blocks: its low half is the first block's data, its
// high half the second's. Pairing the low (or high) halves of two taps' shifts
// fills both halves of a vmac.f.
//
// COUNTED promises the pair loop at least two trips, so it pipelines; a short
// row takes the other instance.
template <int K, bool BIAS, bool COUNTED>
__attribute__((noinline)) static void
dwconv1d_cf_blocks(const bfloat16 *restrict in_pad, const bfloat16 *restrict w,
                   bfloat16 *restrict out, int32_t nb) {
  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  ::aie::accum<accfloat, 16> bias_acc;
  bias_acc.from_vector(::aie::broadcast<float, 16>(bias));
  const v16accfloat acc0 = bias_acc;

  // Tap pair j: w[2j] across the low 16 lanes, w[2j + 1] (0 past K) the high.
  constexpr int NP = (K + 1) / 2;
  const ::aie::vector<bfloat16, 16> zero16 = ::aie::zeros<bfloat16, 16>();
  v32bfloat16 coeff[NP];
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < NP; j++)
    coeff[j] = ::aie::concat(
        ::aie::broadcast<bfloat16, 16>(w[2 * j]),
        2 * j + 1 < K ? ::aie::broadcast<bfloat16, 16>(w[2 * j + 1]) : zero16);

  auto two_blocks = [&]() __attribute__((always_inline)) {
    // Samples t .. t + 47; the lanes past t + 47 are never read.
    const ::aie::vector<bfloat16, 32> w0 = ::aie::concat(
        ::aie::load_v<16>(in_pad), ::aie::load_v<16>(in_pad + 16));
    const ::aie::vector<bfloat16, 32> w1 =
        ::aie::load_v<16>(in_pad + 32).template grow<32>();
    in_pad += 32;
    v16accfloat a = acc0, b = acc0;
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < NP; j++) {
      const v32bfloat16 sp = shifted<K>(w0, w1, 2 * j);
      const v32bfloat16 sq = shifted<K>(w0, w1, 2 * j + 1);
      a = mac_elem_16_2(coeff[j], shuffle(sp, sq, INTLV_lo_256o512), a);
      b = mac_elem_16_2(coeff[j], shuffle(sp, sq, INTLV_hi_256o512), b);
    }
    ::aie::store_v(out, ::aie::accum<accfloat, 16>(a).to_vector<bfloat16>());
    ::aie::store_v(out + 16,
                   ::aie::accum<accfloat, 16>(b).to_vector<bfloat16>());
    out += 32;
  };

  const int32_t np = nb / 2;
  if constexpr (COUNTED) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int32_t n = 0; n < np; n++)
      two_blocks();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int32_t n = 0; n < np; n++)
      two_blocks();
  }
  if (nb & 1) {
    const ::aie::vector<bfloat16, 32> w0 = ::aie::concat(
        ::aie::load_v<16>(in_pad), ::aie::load_v<16>(in_pad + 16));
    v16accfloat a = acc0;
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < NP; j++) {
      const v32bfloat16 sp = shifted<K>(w0, w0, 2 * j);
      const v32bfloat16 sq = shifted<K>(w0, w0, 2 * j + 1);
      a = mac_elem_16_2(coeff[j], shuffle(sp, sq, INTLV_lo_256o512), a);
    }
    ::aie::store_v(out, ::aie::accum<accfloat, 16>(a).to_vector<bfloat16>());
  }
}

template <int K, bool BIAS>
static inline void dwconv1d_channels_first_impl(const bfloat16 *restrict in_pad,
                                                const bfloat16 *restrict w,
                                                bfloat16 *restrict out,
                                                int32_t T) {
  static_assert(K >= 1 && K <= 17,
                "K taps must fit one 32-lane window (16 + K - 1 <= 32)");
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const int32_t nb = (uint32_t)(T + 15) / 16;
  if (nb >= 4)
    dwconv1d_cf_blocks<K, BIAS, true>(in_pad, w, out, nb);
  else
    dwconv1d_cf_blocks<K, BIAS, false>(in_pad, w, out, nb);
  ::aie::set_rounding(saved_rounding);
  event1();
}
#elif AIE_TUNED_AIE2P
__attribute__((always_inline)) static inline ::aie::vector<bfloat16, 32>
shifted(const ::aie::vector<bfloat16, 32> &w0,
        const ::aie::vector<bfloat16, 32> &w1, int p) {
  if (p == 0)
    return w0;
  return ::aie::shuffle_down_fill(w0, w1, p);
}

// 32 outputs per block, one shift and one 32-lane vmac.f per tap. The taps
// split into the same two chains, in the same order, as the generic branch so
// the sums round identically.
template <int K>
__attribute__((always_inline)) static inline ::aie::accum<accfloat, 32>
dwconv1d_cf_block(const ::aie::vector<bfloat16, 32> (&taps)[K],
                  const ::aie::vector<float, 32> &bias_v,
                  const ::aie::vector<bfloat16, 32> &w0,
                  const ::aie::vector<bfloat16, 32> &w1) {
  constexpr int KA = (K + 1) / 2;
  ::aie::accum<accfloat, 32> a;
  a.from_vector(bias_v);
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < KA; p++)
    a = ::aie::mac(a, taps[p], shifted(w0, w1, p));
  if constexpr (K > KA) {
    ::aie::accum<accfloat, 32> b = ::aie::mul(taps[KA], shifted(w0, w1, KA));
    AIE_LOOP_UNROLL_FULL
    for (int p = KA + 1; p < K; p++)
      b = ::aie::mac(b, taps[p], shifted(w0, w1, p));
    a = ::aie::add(a, b);
  }
  return a;
}

template <int K, bool BIAS>
static inline void dwconv1d_channels_first_impl(const bfloat16 *restrict in_pad,
                                                const bfloat16 *restrict w,
                                                bfloat16 *restrict out,
                                                int32_t T) {
  static_assert(K >= 1 && K <= 17,
                "K taps must fit one 32-lane window (16 + K - 1 <= 32)");
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<float, 32> bias_v = ::aie::broadcast<float, 32>(bias);
  ::aie::vector<bfloat16, 32> taps[K];
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < K; p++)
    taps[p] = ::aie::broadcast<bfloat16, 32>(w[p]);

  auto block32 = [&]() __attribute__((always_inline)) {
    // Samples t .. t + 47; the lanes past t + 47 are never read.
    const ::aie::vector<bfloat16, 32> w0 = ::aie::concat(
        ::aie::load_v<16>(in_pad), ::aie::load_v<16>(in_pad + 16));
    const ::aie::vector<bfloat16, 32> w1 =
        ::aie::load_v<16>(in_pad + 32).template grow<32>();
    in_pad += 32;
    ::aie::store_v(out, dwconv1d_cf_block<K>(taps, bias_v, w0, w1)
                            .template to_vector<bfloat16>());
    out += 32;
  };
  // Four promised trips let the pipeliner overlap blocks; a short row takes
  // the plain loop.
  const int32_t nb = (uint32_t)T / 32;
  if (nb >= 4) {
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int32_t n = nb; n > 0; n--)
      block32();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int32_t n = nb; n > 0; n--)
      block32();
  }
  if (T & 16) {
    // Only the low 16 lanes are kept; they read samples t .. t + 31.
    const ::aie::vector<bfloat16, 32> w0 = ::aie::concat(
        ::aie::load_v<16>(in_pad), ::aie::load_v<16>(in_pad + 16));
    ::aie::store_v(out, dwconv1d_cf_block<K>(taps, bias_v, w0, w0)
                            .template to_vector<bfloat16>()
                            .template extract<16>(0));
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}
#else
template <int K, bool BIAS>
static inline void dwconv1d_channels_first_impl(const bfloat16 *restrict in_pad,
                                                const bfloat16 *restrict w,
                                                bfloat16 *restrict out,
                                                int32_t T) {
  static_assert(K >= 1 && K <= 17,
                "K taps must fit one 32-lane window (16 + K - 1 <= 32)");
  event0();
  // Rounding is one sticky register shared by every kernel on this core.
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<float, 16> bias_v = ::aie::broadcast<float, 16>(bias);

  // sliding_mul indexes the coefficients modulo the vector length.
  constexpr unsigned kCoeffLanes = K <= 16 ? 16 : 32;
  ::aie::vector<bfloat16, kCoeffLanes> taps =
      ::aie::zeros<bfloat16, kCoeffLanes>();
  // Unrolled: with a running p, taps.set is a memory round-trip.
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < K; p++)
    taps.set(w[p], p);

  // Two independent half-length chains rather than K dependent vmac.f.
  constexpr int KA = (K + 1) / 2;
  constexpr int KB = K - KA;
  using conv_a = ::aie::sliding_mul_ops<16, KA, 1, 1, 1, bfloat16, bfloat16>;
  using conv_b =
      ::aie::sliding_mul_ops<16, KB ? KB : 1, 1, 1, 1, bfloat16, bfloat16>;

  // A down-count over walking cursors for the zero-overhead loop; unsigned so
  // the block count is a shift. Two blocks per pass interleave their chains.
  AIE_LOOP_UNROLL(2)
  for (int32_t n = (uint32_t)(T + 15) / 16; n > 0; n--) {
    // in_pad is only 256-bit aligned; a 512-bit access needs 512-bit alignment.
    const ::aie::vector<bfloat16, 32> window = ::aie::concat(
        ::aie::load_v<16>(in_pad), ::aie::load_v<16>(in_pad + 16));
    in_pad += 16;
    ::aie::accum<accfloat, 16> acc;
    acc.from_vector(bias_v);
    auto acc_a = conv_a::mac(acc, taps, 0, window, 0);
    if constexpr (KB > 0) {
      auto acc_b = conv_b::mul(taps, KA, window, KA);
      ::aie::store_v(out,
                     ::aie::add(acc_a, acc_b).template to_vector<bfloat16>());
    } else {
      ::aie::store_v(out, acc_a.template to_vector<bfloat16>());
    }
    out += 16;
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}
#endif

#ifndef DWCONV1D_CF_K
#define DWCONV1D_CF_K 9
#endif
#ifndef DWCONV1D_CF_BIAS
#define DWCONV1D_CF_BIAS 1
#endif

extern "C" {

// w holds taps [0 .. DWCONV1D_CF_K-1] with the bias at [DWCONV1D_CF_K]. A
// caller may pass a wider row (dwconv1d.py pads for 4-byte aie.dma_bd
// alignment); anything past the bias is never read.
void dwconv1d_channels_first_bf16(bfloat16 *in_pad, bfloat16 *w, bfloat16 *out,
                                  int32_t T) {
  dwconv1d_channels_first_impl<DWCONV1D_CF_K, (bool)DWCONV1D_CF_BIAS>(in_pad, w,
                                                                      out, T);
}

} // extern "C"
