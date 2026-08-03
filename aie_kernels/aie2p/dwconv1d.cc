//===- dwconv1d.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Depthwise conv1d, 'same' padding, stride 1, bf16, one channel per call.
// Cross-correlation, no kernel flip, matching torch.nn.Conv1d.
//
// The caller supplies the padded row [P zeros | T samples | P zeros | slack],
// P = (K-1)/2, with a fixed 16 elements of slack whatever K is so the aligned
// 16-wide loads never read past the buffer. in_pad must be 256-bit aligned and
// T a multiple of 16; dwconv1d.py's `_pad_input` builds one.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

template <int K, bool BIAS>
static inline void dwconv1d_same_bf16_impl(const bfloat16 *restrict in_pad,
                                           const bfloat16 *restrict w,
                                           bfloat16 *restrict out, int32_t T) {
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

#ifndef DWCONV_K
#define DWCONV_K 9
#endif
#ifndef DWCONV_BIAS
#define DWCONV_BIAS 1
#endif

extern "C" {

// w holds taps [0 .. DWCONV_K-1] with the bias at [DWCONV_K]. A caller may pass
// a wider row (dwconv1d.py pads for 4-byte aie.dma_bd alignment); anything past
// the bias is never read.
void dwconv1d_bf16(bfloat16 *in_pad, bfloat16 *w, bfloat16 *out, int32_t T) {
  dwconv1d_same_bf16_impl<DWCONV_K, (bool)DWCONV_BIAS>(in_pad, w, out, T);
}

} // extern "C"
