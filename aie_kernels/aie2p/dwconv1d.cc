//===- dwconv1d.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Depthwise conv1d, 'same' padding, stride 1, bf16, one channel (row) per
// call:
//
//   out[t] = bias + sum_{p=0..K-1} w[p] * in_pad[t + p],   t = 0 .. T-1
//
// A cross-correlation, no kernel flip, matching torch.nn.Conv1d and most
// framework "same" depthwise convs. The caller supplies the padded row, so
// the kernel does no boundary handling and needs no scratch buffer:
//
//   in_pad[0 .. T+15]: [P zeros | T real samples | P zeros | 16-(K-1) junk]
//                       \_______________ T+K-1 valid, halo-padded ________/
//   P = (K - 1) / 2
//
// The slack past the halo is a fixed 16 elements whatever K is, so that the
// aligned 16-wide loads never read past the end of the buffer; its values
// never reach the output. dwconv1d.py's `_pad_input` builds one.
//
// K is compile-time (DWCONV_K, default 9). The 16 + K - 1 wide shuffle window
// must fit the 32-lane two-register concat, so K <= 17, and in_pad must be
// aligned with T a multiple of 16.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

template <int K, bool BIAS>
static inline void dwconv1d_same_bf16_impl(const bfloat16 *restrict in_pad,
                                           const bfloat16 *restrict w,
                                           bfloat16 *restrict out, int32_t T) {
  static_assert(
      K >= 1 && K <= 17,
      "K taps must fit one 32-lane shuffle window (16 + K - 1 <= 32)");
  event0();
  // The rounding mode is one sticky register shared by every kernel on this
  // core, so conv_even must be handed back before returning.
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<float, 16> bias_v = ::aie::broadcast<float, 16>(bias);

  // One broadcast per tap, hoisted out of the T-loop.
  ::aie::vector<bfloat16, 16> coeff[K];
  for (int p = 0; p < K; p++)
    coeff[p] = ::aie::broadcast<bfloat16, 16>(w[p]);

  for (int32_t o = 0; o < T; o += 16) {
    const ::aie::vector<bfloat16, 16> a0 = ::aie::load_v<16>(in_pad + o);
    const ::aie::vector<bfloat16, 16> a1 = ::aie::load_v<16>(in_pad + o + 16);
    ::aie::accum<accfloat, 16> acc;
    acc.from_vector(bias_v);
    for (int p = 0; p < K; p++) {
      // in_pad[o+p .. o+p+15]. p <= K-1 <= 16 selects only lanes [0, p) of
      // a1, so the don't-care tail never reaches the accumulator.
      const ::aie::vector<bfloat16, 16> window =
          ::aie::shuffle_down_fill(a0, a1, static_cast<unsigned>(p));
      acc = ::aie::mac(acc, window, coeff[p]);
    }
    ::aie::store_v(out + o, acc.template to_vector<bfloat16>());
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

//   in_pad: T + 16 bf16, halo-padded as in the file header
//   w:      taps [0 .. DWCONV_K-1], bias at [DWCONV_K] if DWCONV_BIAS. A
//           caller may pass a wider row (dwconv1d.py pads to keep the
//           aie.dma_bd transfer length 4-byte aligned); anything past the
//           bias is never read.
//   out:    T bf16
//   T:      output length, must be a multiple of 16
void dwconv1d_bf16(bfloat16 *in_pad, bfloat16 *w, bfloat16 *out, int32_t T) {
  dwconv1d_same_bf16_impl<DWCONV_K, (bool)DWCONV_BIAS>(in_pad, w, out, T);
}

} // extern "C"
