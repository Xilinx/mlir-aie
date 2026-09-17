//===- dwconv1d_channels_first.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

// Depthwise conv1d over a channels-first tensor: one channel per call, time
// contiguous, 'same' padding, stride 1, bf16. Cross-correlation with no kernel
// flip, matching torch.nn.Conv1d. Taps are K scalars and the vectorization runs
// along time, via sliding_mul.
//
// This is the general one: runtime T, 'same' padding, optional bias. See
// dwconv1d_channels_last.cc for the transposed layout, and its header for the
// throughput comparison and which to reach for.
//
// The caller supplies the padded row [P zeros | T samples | P zeros | slack],
// P = (K-1)/2, with a fixed 16 elements of slack whatever K is so the aligned
// 16-wide loads never read past the buffer. in_pad must be 256-bit aligned and
// T a multiple of 16; dwconv1d.py's `_pad_input` builds one.

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
  for (int p = 0; p < K; p++)
    taps.set(w[p], p);

  using conv = ::aie::sliding_mul_ops<16, K, 1, 1, 1, bfloat16, bfloat16>;

  for (int32_t o = 0; o < T; o += 16) {
    // in_pad is only 256-bit aligned; a 512-bit access needs 512-bit alignment.
    const ::aie::vector<bfloat16, 32> window = ::aie::concat(
        ::aie::load_v<16>(in_pad + o), ::aie::load_v<16>(in_pad + o + 16));
    ::aie::accum<accfloat, 16> acc;
    acc.from_vector(bias_v);
    ::aie::store_v(
        out + o,
        conv::mac(acc, taps, 0, window, 0).template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

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
