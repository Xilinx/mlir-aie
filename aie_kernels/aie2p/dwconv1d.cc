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
// K is compile-time (DWCONV_K, default 9). The window `sliding_mul_ops` reads
// is 16 + K - 1 wide and must fit one 32-lane vector, so K <= 17. in_pad must
// be 256-bit aligned and T a multiple of 16.
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
  // The rounding mode is one sticky register shared by every kernel on this
  // core, so conv_even must be handed back before returning.
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<float, 16> bias_v = ::aie::broadcast<float, 16>(bias);

  // sliding_mul indexes the coefficient vector modulo its length, so it has to
  // be wide enough to hold all K taps distinctly.
  constexpr unsigned kCoeffLanes = K <= 16 ? 16 : 32;
  ::aie::vector<bfloat16, kCoeffLanes> taps =
      ::aie::zeros<bfloat16, kCoeffLanes>();
  for (int p = 0; p < K; p++)
    taps.set(w[p], p);

  using conv = ::aie::sliding_mul_ops<16, K, 1, 1, 1, bfloat16, bfloat16>;

  for (int32_t o = 0; o < T; o += 16) {
    // Two 256-bit loads rather than one 512-bit load: in_pad is only 256-bit
    // aligned, and a 512-bit access on XDNA 2 requires 512-bit alignment.
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
