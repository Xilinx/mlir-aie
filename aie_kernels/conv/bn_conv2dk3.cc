//===- conv2dk3.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
// #define __AIENGINE__ 2
#define NOCPP
// #define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_arch.h"
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

enum region { top, middle, bottom };

const int32_t MAX = 255;

//*****************************************************************************
// conv2d 3x3 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

void conv2dk3_i8_stride2_scalar(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts, uint8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t check, const int scale,
    const int channel_offset) {
  event0();

  int x, ki, ic, oc, ic8, oc8;
  int32_t sum;
  int sum_srs;
  int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
  int in_indx_0 = 0;
  int output_width = input_width / 2; // Stride 2 reduces output width by half

  for (oc = 0; oc < output_channels / 8; oc++) {
    int oc_ofst = oc + (channel_offset / 8);
    for (oc8 = 0; oc8 < 8; oc8++) {

      // left border
      sum = 0;
      sum_srs = 0;
      for (ic = 0; ic < input_channels / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          for (ki = 1; ki < kernel_width; ki++) {
            int wts_indx_0 =
                (0 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                (ic8 * 8) +
                (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
            int wts_indx_1 =
                (1 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                (ic8 * 8) +
                (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
            int wts_indx_2 =
                (2 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                (ic8 * 8) +
                (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;

            in_indx_0 = (0 + ki - 1) * 8 + ((ic * input_width * 8) + ic8);

            if (check != top)
              sum += line0[in_indx_0] * wts[wts_indx_0];
            sum += line1[in_indx_0] * wts[wts_indx_1];
            if (check != bottom)
              sum += line2[in_indx_0] * wts[wts_indx_2];
          }
        }
      }
      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs =
          (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
      output[(oc * output_width * 8) + oc8] = sum_srs;

      for (x = 1; x < output_width;
           x++) { // stride 2 means we skip one input column
        sum = 0;
        sum_srs = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            for (ki = 0; ki < kernel_width; ki++) {
              int wts_indx_0 =
                  (0 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                  (ic8 * 8) +
                  (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) +
                  oc8;
              int wts_indx_1 =
                  (1 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                  (ic8 * 8) +
                  (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) +
                  oc8;
              int wts_indx_2 =
                  (2 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) +
                  (ic8 * 8) +
                  (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) +
                  oc8;

              int in_indx_0 =
                  (2 * x - 1 + ki) * 8 + ((ic * input_width * 8) + ic8);

              if (check != top)
                sum += line0[in_indx_0] * wts[wts_indx_0];
              sum += line1[in_indx_0] * wts[wts_indx_1];
              if (check != bottom)
                sum += line2[in_indx_0] * wts[wts_indx_2];
            }
          }
        }
        // sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >>
                   scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * output_width * 8) + x * 8 + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
// AIE2P keeps these short loops rolled and their register arrays on the
// stack unless told to unroll them; the AIE2 build is left as tuned.
#if AIE_TUNED_AIE2P
#define BN3_UNROLL_FULL AIE_LOOP_UNROLL_FULL
#else
#define BN3_UNROLL_FULL
#endif

// On AIE2P the factory passes the shape, which keeps only the code that shape
// takes; the matching arguments are then not read.
#if AIE_TUNED_AIE2P && defined(CONV_INPUT_WIDTH)
#define K3_SHAPE(arg, flag) (flag)
#else
#define K3_SHAPE(arg, flag) (arg)
#endif

// The line buffers and weights are read 64 bytes at a time, which AIE2P
// rounds down to a 64-byte boundary; the output is stored 32 bytes at a time.
#if AIE_TUNED_AIE2P
#define K3_RESTRICT __restrict
#define K3_LOAD_ALIGN 64
#define K3_MISALIGNED(l0, l1, l2, w, o)                                        \
  ((((uintptr_t)(l0) | (uintptr_t)(l1) | (uintptr_t)(l2) | (uintptr_t)(w)) &   \
    63) |                                                                      \
   ((uintptr_t)(o) & 31))
#else
#define K3_RESTRICT
#define K3_LOAD_ALIGN 32
#define K3_MISALIGNED(l0, l1, l2, w, o)                                        \
  (((uintptr_t)(l0) | (uintptr_t)(l1) | (uintptr_t)(l2) | (uintptr_t)(w) |     \
    (uintptr_t)(o)) &                                                          \
   31)
#endif

// Stride-2 3x3, input_width a multiple of 8; see k1_load in
// bn_conv2dk1_aie2.h for the layout and mmul tiling.
// Input pixels 2x .. 2x + 7 split with filter_even into the centre tap and
// filter_odd into the right tap; the odd pixels shifted up by one, with pixel
// 2x - 1 from the chunk before (zero at x = 0), give the left tap. Rows
// dropped by `check` are skipped.
template <int N, bool Left>
static inline void k3_chunks(const int8_t *const *lines, const int8_t *wts,
                             uint8_t *__restrict out, const int32_t r0,
                             const int32_t r1, const int32_t row,
                             const int32_t ic_blocks, const int scale) {
  using MMUL = aie::mmul<4, 8, 8, int8, int8>;
  MMUL acc[N];
  BN3_UNROLL_FULL
  for (int j = 0; j < N; j++)
    acc[j] = MMUL(aie::zeros<acc32, 32>());
  for (int r = r0; r < r1; r++) {
    const int8_t *in = lines[r];
    const int8_t *w = wts + r * 192;
#pragma clang loop min_iteration_count(1)
    for (int ic = 0; ic < ic_blocks; ic++) {
      aie::vector<int8, 32> prev;
      if constexpr (Left)
        prev = aie::zeros<int8, 32>();
      else
        prev = aie::load_v<32>(in - 32);
      const aie::vector<int8, 64> b0 = aie::load_v<64>(w);
      const aie::vector<int8, 64> b1 = aie::load_v<64>(w + 64);
      const aie::vector<int8, 64> b2 = aie::load_v<64>(w + 128);
      BN3_UNROLL_FULL
      for (int j = 0; j < N; j++) {
        const aie::vector<int8, 64> v = aie::load_v<64>(in + 64 * j);
        const aie::vector<int8, 32> c = aie::filter_even(v, 8);
        const aie::vector<int8, 32> rt = aie::filter_odd(v, 8);
        acc[j].mac(aie::shuffle_up_fill(rt, prev, 8), b0);
        acc[j].mac(c, b1);
        acc[j].mac(rt, b2);
        prev = rt;
      }
      in += row;
      w += 576;
    }
  }
  BN3_UNROLL_FULL
  for (int j = 0; j < N; j++)
    aie::store_v(out + 32 * j, acc[j].template to_vector<uint8>(scale));
}

// input_channels == 8: one pass over the row per output channel block, each
// row's odd pixels carried from chunk to chunk. A row dropped by `check` gets
// zero weights.
alignas(K3_LOAD_ALIGN) static const int8_t k3_zero_wts[3 * 64] = {};

template <int N>
static inline void
k3_ic1_step(const int8_t *const *lines, const int8_t *const *wr,
            aie::vector<int8, 32> *prev, uint8_t *out, const int scale) {
  using MMUL = aie::mmul<4, 8, 8, int8, int8>;
  MMUL acc[N];
#pragma unroll
  for (int r = 0; r < 3; r++) {
    const aie::vector<int8, 64> b0 = aie::load_v<64>(wr[r]);
    const aie::vector<int8, 64> b1 = aie::load_v<64>(wr[r] + 64);
    const aie::vector<int8, 64> b2 = aie::load_v<64>(wr[r] + 128);
    BN3_UNROLL_FULL
    for (int j = 0; j < N; j++) {
      const aie::vector<int8, 64> v = aie::load_v<64>(lines[r] + 64 * j);
      const aie::vector<int8, 32> c = aie::filter_even(v, 8);
      const aie::vector<int8, 32> rt = aie::filter_odd(v, 8);
      if (r == 0)
        acc[j].mul(aie::shuffle_up_fill(rt, prev[r], 8), b0);
      else
        acc[j].mac(aie::shuffle_up_fill(rt, prev[r], 8), b0);
      acc[j].mac(c, b1);
      acc[j].mac(rt, b2);
      prev[r] = rt;
    }
  }
  BN3_UNROLL_FULL
  for (int j = 0; j < N; j++)
    aie::store_v(out + 32 * j, acc[j].template to_vector<uint8>(scale));
}

#if AIE_TUNED_AIE2P
// Eight output pixels per mac: each row's 16 input pixels split into the
// centre and right taps in one shuffle each. Half: the last four pixels only.
// Split: out is 32 bytes off a 64-byte boundary, where a 512-bit store would
// round down.
template <int N, bool Split, bool Half = false>
static inline void
k3_ic1_step8(const int8_t *const *lines, const int8_t *const *wr,
             aie::vector<int8, 64> *prev, uint8_t *out, const int scale) {
  aie::mmul<8, 8, 8, int8, int8> acc[N];
#pragma unroll
  for (int r = 0; r < 3; r++) {
    const aie::vector<int8, 64> b0 = aie::load_v<64>(wr[r]);
    const aie::vector<int8, 64> b1 = aie::load_v<64>(wr[r] + 64);
    const aie::vector<int8, 64> b2 = aie::load_v<64>(wr[r] + 128);
    BN3_UNROLL_FULL
    for (int j = 0; j < N; j++) {
      const aie::vector<int8, 128> v =
          aie::concat(aie::load_v<64>(lines[r] + 128 * j),
                      Half ? aie::zeros<int8, 64>()
                           : aie::load_v<64>(lines[r] + 128 * j + 64));
      const aie::vector<int8, 64> c = aie::filter_even(v, 8);
      const aie::vector<int8, 64> rt = aie::filter_odd(v, 8);
      const aie::vector<int8, 64> lt = aie::shuffle_up_fill(rt, prev[r], 8);
      if (r == 0)
        acc[j].mul(lt, b0);
      else
        acc[j].mac(lt, b0);
      acc[j].mac(c, b1);
      acc[j].mac(rt, b2);
      prev[r] = rt;
    }
  }
  BN3_UNROLL_FULL
  for (int j = 0; j < N; j++) {
    const aie::vector<uint8, 64> o = acc[j].template to_vector<uint8>(scale);
    if constexpr (Half || Split) {
      aie::store_v(out + 64 * j, o.extract<32>(0));
      if constexpr (!Half)
        aie::store_v(out + 64 * j + 32, o.extract<32>(1));
    } else {
      aie::store_v(out + 64 * j, o);
    }
  }
}

template <bool Split>
static void
k3_ic1_rows8(const int8_t *__restrict line0, const int8_t *__restrict line1,
             const int8_t *__restrict line2, const int8_t *__restrict wts,
             uint8_t *__restrict output, const int32_t output_width,
             const int32_t output_channels, const int32_t check,
             const int scale, const int channel_offset) {
  const int32_t chunks = output_width / 4;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *w = wts + (oc + channel_offset / 8) * 3 * 3 * 64;
    const int8_t *wr[3] = {check == top ? k3_zero_wts : w, w + 192,
                           check == bottom ? k3_zero_wts : w + 384};
    const int8_t *l[3] = {line0, line1, line2};
    uint8_t *out = output + oc * output_width * 8;
    aie::vector<int8, 64> prev[3] = {
        aie::zeros<int8, 64>(), aie::zeros<int8, 64>(), aie::zeros<int8, 64>()};
    int x = 0;
    // Run just once, this loop's loads would be hoisted out of the oc loop
    // and spill (1216 B of stack at 56 pixels).
    if (chunks >= 8)
      for (; x + 4 <= chunks; x += 4) {
        k3_ic1_step8<2, Split>(l, wr, prev, out, scale);
        BN3_UNROLL_FULL
        for (int i = 0; i < 3; i++)
          l[i] += 256;
        out += 128;
      }
    for (; x + 2 <= chunks; x += 2) {
      k3_ic1_step8<1, Split>(l, wr, prev, out, scale);
      BN3_UNROLL_FULL
      for (int i = 0; i < 3; i++)
        l[i] += 128;
      out += 64;
    }
    if (chunks & 1)
      k3_ic1_step8<1, Split, true>(l, wr, prev, out, scale);
  }
}
#endif

static void k3_ic1_rows(const int8_t *K3_RESTRICT line0,
                        const int8_t *K3_RESTRICT line1,
                        const int8_t *K3_RESTRICT line2,
                        const int8_t *K3_RESTRICT wts,
                        uint8_t *K3_RESTRICT output, const int32_t input_width,
                        const int32_t output_channels, const int32_t check,
                        const int scale, const int channel_offset) {
  const int32_t output_width = input_width / 2;
#if AIE_TUNED_AIE2P
  // Odd chunks leave every other output row 32 bytes off a 64-byte boundary.
  if ((output_width / 4) % 2 == 0 && ((uintptr_t)output & 63) == 0)
    k3_ic1_rows8<false>(line0, line1, line2, wts, output, output_width,
                        output_channels, check, scale, channel_offset);
  else
    k3_ic1_rows8<true>(line0, line1, line2, wts, output, output_width,
                       output_channels, check, scale, channel_offset);
#else
  const int32_t chunks = output_width / 4;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *w = wts + (oc + channel_offset / 8) * 3 * 3 * 64;
    const int8_t *wr[3] = {check == top ? k3_zero_wts : w, w + 192,
                           check == bottom ? k3_zero_wts : w + 384};
    const int8_t *l[3] = {line0, line1, line2};
    uint8_t *out = output + oc * output_width * 8;
    aie::vector<int8, 32> prev[3] = {
        aie::zeros<int8, 32>(), aie::zeros<int8, 32>(), aie::zeros<int8, 32>()};
    for (int x = 0; x + 2 <= chunks; x += 2) {
      k3_ic1_step<2>(l, wr, prev, out, scale);
      BN3_UNROLL_FULL
      for (int i = 0; i < 3; i++)
        l[i] += 128;
      out += 64;
    }
    if (chunks & 1)
      k3_ic1_step<1>(l, wr, prev, out, scale);
  }
#endif
}

static void k3_stride2_vector(const int8_t *line0, const int8_t *line1,
                              const int8_t *line2, const int8_t *wts,
                              uint8_t *output, const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels,
                              const int32_t check, const int scale,
                              const int channel_offset) {
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (input_channels == 8) {
    k3_ic1_rows(line0, line1, line2, wts, output, input_width, output_channels,
                check, scale, channel_offset);
    event1();
    return;
  }
  constexpr int N = 4;
  const int32_t output_width = input_width / 2;
  const int32_t row = input_width * 8;
  const int32_t ic_blocks = input_channels / 8;
  const int32_t groups = (output_width / 4 - 1) / N;
  const int32_t rem = (output_width / 4 - 1) % N;
  const int32_t r0 = check == top ? 1 : 0;
  const int32_t r1 = check == bottom ? 2 : 3;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *w = wts + (oc + channel_offset / 8) * ic_blocks * 3 * 3 * 64;
    uint8_t *out = output + oc * output_width * 8;
    const int8_t *l[3] = {line0, line1, line2};
    k3_chunks<1, true>(l, w, out, r0, r1, row, ic_blocks, scale);
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++)
      l[i] += 64;
    out += 32;
    for (int g = 0; g < groups; g++) {
      k3_chunks<N, false>(l, w, out, r0, r1, row, ic_blocks, scale);
      BN3_UNROLL_FULL
      for (int i = 0; i < 3; i++)
        l[i] += 64 * N;
      out += 32 * N;
    }
    switch (rem) {
    case 1:
      k3_chunks<1, false>(l, w, out, r0, r1, row, ic_blocks, scale);
      break;
    case 2:
      k3_chunks<2, false>(l, w, out, r0, r1, row, ic_blocks, scale);
      break;
    case 3:
      k3_chunks<3, false>(l, w, out, r0, r1, row, ic_blocks, scale);
      break;
    }
  }
  event1();
}
#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

extern "C" {

void conv2dk3_stride2_i8(int8_t *line0, int8_t *line1, int8_t *line2,
                         int8_t *wts, uint8_t *output,
                         const int32_t input_width,
                         const int32_t input_channels,
                         const int32_t output_channels,
                         const int32_t kernel_width,
                         const int32_t kernel_height, const int32_t check,
                         const int scale, const int channel_offset) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  const int32_t width = K3_SHAPE(input_width, CONV_INPUT_WIDTH);
  if (kernel_width == 3 && width >= 8 && width % 8 == 0 &&
      K3_MISALIGNED(line0, line1, line2, wts, output) == 0) {
    k3_stride2_vector(line0, line1, line2, wts, output, width,
                      K3_SHAPE(input_channels, CONV_INPUT_CHANNELS),
                      K3_SHAPE(output_channels, CONV_OUTPUT_CHANNELS), check,
                      scale, channel_offset);
    return;
  }
#endif
  conv2dk3_i8_stride2_scalar(line0, line1, line2, wts, output, input_width,
                             input_channels, output_channels, kernel_width,
                             kernel_height, check, scale, channel_offset);
}
}