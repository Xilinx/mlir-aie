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
#include <type_traits>
#include <utility>

#include "../aie_arch.h"
#include "../aie_kernel_utils.h"
#include "../common/store.h"
#include "../common/table_i8.h"
#include "conv_bias.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

enum region { top, middle, bottom };

// On AIE2P the factories pass the row width and channel count, which keeps
// only the code those take; the matching arguments are then not read.
#if AIE_TUNED_AIE2P && defined(CONV_INPUT_WIDTH)
#define DW_WIDTH(w) CONV_INPUT_WIDTH
#else
#define DW_WIDTH(w) (w)
#endif
#if AIE_TUNED_AIE2P && defined(CONV_OUTPUT_CHANNELS)
#define DW_CHANNELS(c) CONV_OUTPUT_CHANNELS
#else
#define DW_CHANNELS(c) (c)
#endif

#ifdef SCALAR

const int32_t MAX = 255;

#ifdef STRIDE1_OUT_SPLIT
void conv2dk3_ui8_out_split_scalar(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output1, uint8_t *output2, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t check, const int scale, const int channel_offset) {
  event0();

  int x, ki, c_div_8, c8;
  int32_t sum;
  int32_t sum_srs;
  int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
  int in_indx_0 = 0;
  int CHANNEL_REMAIN = output_channels / 8;
  int VECTOR_SIZE = 8;
  int half_output_channels = output_channels / 2;

  for (c_div_8 = 0; c_div_8 < CHANNEL_REMAIN; c_div_8++) {
    for (c8 = 0; c8 < VECTOR_SIZE; c8++) {
      // Left border
      sum = 0;
      sum_srs = 0;
      for (ki = 1; ki < kernel_width; ki++) {
        wts_indx_0 = 3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        wts_indx_1 = 3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        wts_indx_2 = 3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                    (0 + ki - 1) * VECTOR_SIZE + c8;

        if (check != top)
          sum += line0[in_indx_0] * wts[wts_indx_0];
        sum += line1[in_indx_0] * wts[wts_indx_1];
        if (check != bottom)
          sum += line2[in_indx_0] * wts[wts_indx_2];
      }
      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs =
          (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;

      // Assign to output1 or output2
      if (c_div_8 < half_output_channels / 8) {
        output1[c_div_8 * input_width * VECTOR_SIZE + c8] = sum_srs;
      } else {
        output2[(c_div_8 - half_output_channels / 8) * input_width *
                    VECTOR_SIZE +
                c8] = sum_srs;
      }

      // Right border
      sum = 0;
      sum_srs = 0;
      for (ki = 0; ki < kernel_width - 1; ki++) {
        wts_indx_0 = 3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        wts_indx_1 = 3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        wts_indx_2 = 3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                     ki * VECTOR_SIZE + c8;
        in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                    (input_width - 2 + ki) * VECTOR_SIZE + c8;

        if (check != top)
          sum += line0[in_indx_0] * wts[wts_indx_0];
        sum += line1[in_indx_0] * wts[wts_indx_1];
        if (check != bottom)
          sum += line2[in_indx_0] * wts[wts_indx_2];
      }
      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs =
          (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;

      // Assign to output1 or output2
      if (c_div_8 < half_output_channels / 8) {
        output1[c_div_8 * input_width * VECTOR_SIZE +
                (input_width - 1) * VECTOR_SIZE + c8] = sum_srs;
      } else {
        output2[(c_div_8 - half_output_channels / 8) * input_width *
                    VECTOR_SIZE +
                (input_width - 1) * VECTOR_SIZE + c8] = sum_srs;
      }

      // Middle part of row
      for (x = 1; x < input_width - 1; x++) {
        sum = 0;
        sum_srs = 0;
        for (ki = 0; ki < kernel_width; ki++) {
          wts_indx_0 = 3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                       ki * VECTOR_SIZE + c8;
          wts_indx_1 = 3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                       ki * VECTOR_SIZE + c8;
          wts_indx_2 = 3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                       ki * VECTOR_SIZE + c8;
          in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                      (x - 1 + ki) * VECTOR_SIZE + c8;

          if (check != top)
            sum += line0[in_indx_0] * wts[wts_indx_0];
          sum += line1[in_indx_0] * wts[wts_indx_1];
          if (check != bottom)
            sum += line2[in_indx_0] * wts[wts_indx_2];
        }
        // sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >>
                   scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;

        // Assign to output1 or output2
        if (c_div_8 < half_output_channels / 8) {
          output1[c_div_8 * input_width * VECTOR_SIZE + x * VECTOR_SIZE + c8] =
              sum_srs;
        } else {
          output2[(c_div_8 - half_output_channels / 8) * input_width *
                      VECTOR_SIZE +
                  x * VECTOR_SIZE + c8] = sum_srs;
        }
      }
    }
  }

  event1();
}
#endif // STRIDE1_OUT_SPLIT

//*****************************************************************************
// conv2d 3x3 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
#ifdef STRIDE2

static void conv2dk3_stride2_ui8_scalar(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t check, const int scale,
    const int channel_offset) {
  event0();

  int x, ki, c_div_8, c8;
  int32_t sum;
  int32_t sum_srs;
  int remain;
  int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
  int in_indx_0 = 0;
  int32_t output_width = input_width / 2;
  // for (oc = (0+channel_offset)/8; oc < (output_channels+channel_offset)/8;
  // oc++) {
  int CHANNEL_REMAIN = output_channels / 8;
  int VECTOR_SIZE = 8;
  for (c_div_8 = 0; c_div_8 < CHANNEL_REMAIN; c_div_8++) {
    for (c8 = 0; c8 < VECTOR_SIZE; c8++) {
      // left border
      sum = 0;
      sum_srs = 0;
      for (ki = 1; ki < kernel_width; ki++) {
        // wts format - orig is oc,ic,ky,kx, reformat is
        // oc,ic,k0..k8,ic8,oc8
        int wts_indx_0 = +3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_1 = +3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_2 = +3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                        (0 + ki - 1) * VECTOR_SIZE + c8;
        if (check != top)
          sum += line0[in_indx_0] * wts[wts_indx_0];
        sum += line1[in_indx_0] * wts[wts_indx_1];
        if (check != bottom)
          sum += line2[in_indx_0] * wts[wts_indx_2];
      }

      // remain = sum & ((1<<(scale-1))-1); // is there any bit set, not a tie
      // case if (remain > 0){ sum += 1<<(scale-1); } sum_srs = (sum >> scale)
      // << scale;

      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs =
          ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
      output[(c_div_8 * output_width * VECTOR_SIZE) + c8] = sum_srs;

      for (x = 1; x < output_width; x++) { // middle part of row
        sum = 0;
        sum_srs = 0;
        for (ki = 0; ki < kernel_width; ki++) {
          // wts format - orig is oc,ic,ky,kx, reformat is
          // oc,ic,k0..k8,ic8,oc8
          int wts_indx_0 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           0 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int wts_indx_1 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           1 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int wts_indx_2 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           2 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                          (2 * x - 1 + ki) * VECTOR_SIZE + c8;
          if (check != top)
            sum += line0[in_indx_0] * wts[wts_indx_0];
          sum += line1[in_indx_0] * wts[wts_indx_1];
          if (check != bottom)
            sum += line2[in_indx_0] * wts[wts_indx_2];
        }

        // remain = sum & ((1<<(scale-1))-1); // is there any bit set, not a tie
        // case if (remain > 0){ sum += 1<<(scale-1); } sum_srs = (sum >> scale)
        // << scale; sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(c_div_8 * output_width * VECTOR_SIZE) + x * VECTOR_SIZE + c8] =
            sum_srs;
      }
    }
  }

  event1();
}
#else
static void
conv2dk3_ui8_scalar(uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
                    uint8_t *output, const int32_t input_width,
                    const int32_t input_channels, const int32_t output_channels,
                    const int32_t kernel_width, const int32_t kernel_height,
                    const int32_t check, const int scale,
                    const int channel_offset) {
  event0();

  int x, ki, c_div_8, c8;
  int32_t sum;
  int32_t sum_srs;
  int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
  int in_indx_0 = 0;
  // for (oc = (0+channel_offset)/8; oc < (output_channels+channel_offset)/8;
  // oc++) {
  int CHANNEL_REMAIN = output_channels / 8;
  int VECTOR_SIZE = 8;
  for (c_div_8 = 0; c_div_8 < CHANNEL_REMAIN; c_div_8++) {
    for (c8 = 0; c8 < VECTOR_SIZE; c8++) {
      // left border
      sum = 0;
      sum_srs = 0;
      for (ki = 1; ki < kernel_width; ki++) {
        // wts format - orig is oc,ic,ky,kx, reformat is
        // oc,ic,k0..k8,ic8,oc8
        int wts_indx_0 = +3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_1 = +3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_2 = +3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                        (0 + ki - 1) * VECTOR_SIZE + c8;
        if (check != top)
          sum += line0[in_indx_0] * wts[wts_indx_0];
        sum += line1[in_indx_0] * wts[wts_indx_1];
        if (check != bottom)
          sum += line2[in_indx_0] * wts[wts_indx_2];
      }

      sum_srs =
          ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
      output[(c_div_8 * input_width * VECTOR_SIZE) + c8] = sum_srs;

      // right border
      sum = 0;
      sum_srs = 0;
      for (ki = 0; ki < kernel_width - 1; ki++) {
        // wts format - orig is oc,ic,ky,kx, reformat is
        // oc,ic,k0..k8,ic8,oc8
        int wts_indx_0 = +3 * 3 * VECTOR_SIZE * c_div_8 + 0 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_1 = +3 * 3 * VECTOR_SIZE * c_div_8 + 1 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;
        int wts_indx_2 = +3 * 3 * VECTOR_SIZE * c_div_8 + 2 * 3 * VECTOR_SIZE +
                         ki * VECTOR_SIZE + c8;

        int in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                        (input_width - 2 + ki) * VECTOR_SIZE + c8;
        if (check != top)
          sum += line0[in_indx_0] * wts[wts_indx_0];
        sum += line1[in_indx_0] * wts[wts_indx_1];
        if (check != bottom)
          sum += line2[in_indx_0] * wts[wts_indx_2];
      }

      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs =
          ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
      output[(c_div_8 * input_width * VECTOR_SIZE) +
             (input_width - 1) * VECTOR_SIZE + c8] = sum_srs;

      for (x = 1; x < input_width - 1; x++) { // middle part of row
        sum = 0;
        sum_srs = 0;
        for (ki = 0; ki < kernel_width; ki++) {
          // wts format - orig is oc,ic,ky,kx, reformat is
          // oc,ic,k0..k8,ic8,oc8
          int wts_indx_0 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           0 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int wts_indx_1 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           1 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int wts_indx_2 = +3 * 3 * VECTOR_SIZE * c_div_8 +
                           2 * 3 * VECTOR_SIZE + ki * VECTOR_SIZE + c8;
          int in_indx_0 = c_div_8 * input_width * VECTOR_SIZE +
                          (x - 1 + ki) * VECTOR_SIZE + c8;
          if (check != top)
            sum += line0[in_indx_0] * wts[wts_indx_0];
          sum += line1[in_indx_0] * wts[wts_indx_1];
          if (check != bottom)
            sum += line2[in_indx_0] * wts[wts_indx_2];
        }

        // sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(c_div_8 * input_width * VECTOR_SIZE) + x * VECTOR_SIZE + c8] =
            sum_srs;
        // output[oc * (input_width) +  x] = sum;
      }
    }
  }

  event1();
}
#endif // STRIDE
#else  // Vector

#endif // Vector

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
// See K1_UNROLL_CHUNKS in bn_conv2dk1_aie2.h.
#if AIE_TUNED_AIE2P
#define BN3_UNROLL_FULL AIE_LOOP_UNROLL_FULL
#else
#define BN3_UNROLL_FULL
#endif

// One [W][8] row of a channel block per call; 32 lanes are 4 pixels x 8
// channels. A row dropped by `check` gets zero weights, so its line contents
// never reach the sum.
using dw_v = aie::vector<uint8, 32>;
#if AIE_TUNED_AIE2
// AIE2's elementwise multiply sums lanes i and i + 32, so one op takes two
// taps: w[t / 2] holds tap t in half t % 2.
using dw_w = aie::vector<int8, 64>;
constexpr int DW_W = 5;
#else
using dw_w = aie::vector<int8, 32>;
constexpr int DW_W = 9;
#endif

static inline void dw_taps(const int8_t *wts, int32_t check, dw_w *w) {
  BN3_UNROLL_FULL
  for (int r = 0; r < 3; r++) {
    const int32_t m =
        ((r == 0 && check == top) || (r == 2 && check == bottom)) ? 0 : -1;
    BN3_UNROLL_FULL
    for (int ki = 0; ki < 3; ki++) {
      const int32_t *p = (const int32_t *)(wts + (r * 3 + ki) * 8);
      aie::vector<int32, 8> pair = aie::select(
          aie::broadcast<int32, 8>(p[0] & m),
          aie::broadcast<int32, 8>(p[1] & m), aie::mask<8>::from_uint32(0xaa));
#if AIE_TUNED_AIE2
      w[(r * 3 + ki) / 2].insert((r * 3 + ki) % 2,
                                 aie::vector_cast<int8>(pair));
#else
      w[r * 3 + ki] = aie::vector_cast<int8>(pair);
#endif
    }
  }
#if AIE_TUNED_AIE2
  w[4].insert(1, aie::zeros<int8, 32>());
#endif
}

static inline aie::accum<acc32, 32> dw_sum(const dw_v *l, const dw_v *c,
                                           const dw_v *r, const dw_w *w) {
#if AIE_TUNED_AIE2
  const dw_v d[9] = {l[0], c[0], r[0], l[1], c[1], r[1], l[2], c[2], r[2]};
  aie::accum<acc32, 32> acc(::mul_elem_32_2(aie::concat(d[0], d[1]), w[0]));
  for (int i = 1; i < 4; i++)
    acc = ::mac_elem_32_2(aie::concat(d[2 * i], d[2 * i + 1]), w[i], acc);
  return ::mac_elem_32_2(d[8].grow<64>(), w[4], acc);
#else
  aie::accum<acc32, 32> acc = aie::mul(l[0], w[0]);
  acc = aie::mac(acc, c[0], w[1]);
  acc = aie::mac(acc, r[0], w[2]);
  BN3_UNROLL_FULL
  for (int i = 1; i < 3; i++) {
    acc = aie::mac(acc, l[i], w[3 * i]);
    acc = aie::mac(acc, c[i], w[3 * i + 1]);
    acc = aie::mac(acc, r[i], w[3 * i + 2]);
  }
  return acc;
#endif
}

template <bool Aligned, unsigned N>
static inline aie::vector<uint8, N> dw_load(const uint8_t *p) {
  if constexpr (Aligned)
    return aie::load_v<N>(p);
  else
    return aie::load_unaligned_v<N>(p, 8);
}

// See k1_store in bn_conv2dk1_aie2.h.
template <bool Aligned>
static inline void dw_store(uint8_t *p, aie::accum<acc32, 32> acc, int scale) {
  dw_v v = acc.to_vector<uint8>(scale);
  if (Aligned || ((uintptr_t)p & 31) == 0)
    aie::store_v(p, v);
  else
    aie::store_unaligned_v(p, v, 8);
}

// Stride 1, input_width a multiple of 4 and 32-byte aligned rows: each
// chunk's neighbours come from the chunks before and after it.
static void dw_s1_row_aligned(const uint8_t *line0, const uint8_t *line1,
                              const uint8_t *line2, const dw_w *w,
                              uint8_t *__restrict out,
                              const int32_t input_width, const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  dw_v l[3], c[3], r[3], prev[3], next[3];
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    prev[i] = aie::zeros<uint8, 32>();
    c[i] = aie::load_v<32>(in[i]);
  }
  const int last = input_width - 4;
  for (int x = 0; x < last; x += 4) {
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      next[i] = aie::load_v<32>(in[i] + x * 8 + 32);
      l[i] = aie::shuffle_up_fill(c[i], prev[i], 8);
      r[i] = aie::shuffle_down_fill(c[i], next[i], 8);
    }
    dw_store<true>(out + x * 8, dw_sum(l, c, r, w), scale);
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      prev[i] = c[i];
      c[i] = next[i];
    }
  }
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    l[i] = aie::shuffle_up_fill(c[i], prev[i], 8);
    r[i] = aie::shuffle_down_fill(c[i], aie::zeros<uint8, 32>(), 8);
  }
  dw_store<true>(out + last * 8, dw_sum(l, c, r, w), scale);
}

// Stride 1, input_width >= 5: first chunk, 4-pixel middle chunks, and a last
// chunk at input_width - 4, overlapping as bn_conv2dk1_aie2.h's rows do (see
// k1_load there).
static void dw_s1_row(const uint8_t *line0, const uint8_t *line1,
                      const uint8_t *line2, const dw_w *w,
                      uint8_t *__restrict out, const int32_t input_width,
                      const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  dw_v l[3], c[3], r[3];
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    c[i] = dw_load<false, 32>(in[i]);
    r[i] = dw_load<false, 32>(in[i] + 8);
    l[i] = aie::shuffle_up_fill(c[i], aie::zeros<uint8, 32>(), 8);
  }
  dw_store<false>(out, dw_sum(l, c, r, w), scale);

  const int last = input_width - 4;
  for (int x = 4; x < last; x += 4) {
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      l[i] = dw_load<false, 32>(in[i] + (x - 1) * 8);
      c[i] = dw_load<false, 32>(in[i] + x * 8);
      r[i] = dw_load<false, 32>(in[i] + (x + 1) * 8);
    }
    dw_store<false>(out + x * 8, dw_sum(l, c, r, w), scale);
  }

  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    l[i] = dw_load<false, 32>(in[i] + (last - 1) * 8);
    c[i] = dw_load<false, 32>(in[i] + last * 8);
    r[i] = aie::shuffle_down_fill(c[i], aie::zeros<uint8, 32>(), 8);
  }
  dw_store<false>(out + last * 8, dw_sum(l, c, r, w), scale);
}

// Stride 2, output width >= 4: even pixels are the centre tap, odd pixels the
// right tap, and odd pixels shifted up by one the left tap.
template <bool Aligned>
static void dw_s2_row(const uint8_t *line0, const uint8_t *line1,
                      const uint8_t *line2, const dw_w *w,
                      uint8_t *__restrict out, const int32_t output_width,
                      const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  dw_v l[3], c[3], r[3], prev[3];
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++)
    prev[i] = aie::zeros<uint8, 32>();
  const int full = output_width & ~3;
  for (int x = 0; x < full; x += 4) {
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      aie::vector<uint8, 64> v = dw_load<Aligned, 64>(in[i] + x * 16);
      c[i] = aie::filter_even(v, 8);
      r[i] = aie::filter_odd(v, 8);
      l[i] = aie::shuffle_up_fill(r[i], prev[i], 8);
      prev[i] = r[i];
    }
    dw_store<Aligned>(out + x * 8, dw_sum(l, c, r, w), scale);
  }
  if (full != output_width) {
    const int x = output_width - 4;
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      aie::vector<uint8, 64> v = dw_load<false, 64>(in[i] + x * 16);
      c[i] = aie::filter_even(v, 8);
      r[i] = aie::filter_odd(v, 8);
      l[i] = aie::filter_odd(dw_load<false, 64>(in[i] + x * 16 - 16), 8);
    }
    dw_store<false>(out + x * 8, dw_sum(l, c, r, w), scale);
  }
}

#if AIE_TUNED_AIE2P
// AIE2P convolves 8 pixels x 8 channels over 8 taps in one op: output pixel j
// sums tap p times window pixel j + p. With a filter row at taps 0..2 and the
// window starting one pixel before the chunk, each input row is one op per 8
// pixels; of the window's upper 8 pixels only the first two reach the sum.
using dw8_conv = aie::sliding_mul_ch_ops<8, 8, 8, 1, 1, 1, int8, uint8>;
using dw8_v = aie::vector<uint8, 64>;
using dw8_w = aie::vector<int8, 64>;

template <bool Aligned>
static inline void dw8_put(uint8_t *p, dw_v v) {
  if constexpr (Aligned)
    aie::store_v(p, v);
  else
    aie::store_unaligned_v(p, v, 8);
}

// `end` marks a store that ends an output buffer.
template <bool Aligned>
static inline void dw8_put(uint8_t *p, dw_v v, bool end) {
  if constexpr (Aligned)
    aie::store_v(p, v);
  else
    store_unaligned_bounded(p, v, end);
}

template <bool Aligned>
static inline void dw8_store(uint8_t *p, dw8_v v) {
  dw8_put<Aligned>(p, v.extract<32>(0));
  dw8_put<Aligned>(p + 32, v.extract<32>(1));
}

static inline dw8_v dw8_out(aie::accum<acc32, 64> acc, int scale) {
  return acc.to_vector<uint8>(scale);
}

// aie_api's unaligned loads read at the address as given, leaving the hardware
// to round it down (to 32 bytes for a 256-bit load, to 64 for a 512-bit one),
// but tell the compiler it is aligned. The compiler may then take a load's
// data from another load that covers it, which is wrong where the two round
// differently (Xilinx/aie_api#16). Every load here is 512 bits, so a load only
// covers another at the same address.
static inline dw8_v dw8_load(const uint8_t *p) {
  return aie::load_unaligned_v<64>(p, 8);
}

static inline dw8_v dw8_low(dw_v v) {
  return aie::concat(v, aie::zeros<uint8, 32>());
}

// Window of chunk k of a row whose last chunk ends at `end`: from one pixel
// before the chunk, zero outside the row. `keep` is the pixels of a single
// chunk.
template <int Chunks, bool Aligned>
static inline aie::vector<uint8, 128> dw8_window(const uint8_t *row,
                                                 const int k, const int32_t end,
                                                 const aie::mask<64> keep) {
  const dw8_v zero = aie::zeros<uint8, 64>();
  if constexpr (Chunks == 1) {
    const dw8_v u = aie::select(zero, dw8_load(row), keep);
    return aie::concat(aie::shuffle_up_fill(u, zero, 8),
                       aie::shuffle_down_fill(u, zero, 56));
  } else if (k == Chunks - 1) {
    // Two chunks schedule shorter with the upper half zeroed: II 99, not 107.
    if constexpr (Chunks == 2)
      return aie::concat(dw8_load(row + end - 72),
                         dw8_low(aie::shuffle_down_fill(
                             dw8_load(row + end - 64).extract<32>(1),
                             aie::zeros<uint8, 32>(), 24)));
    return aie::concat(
        dw8_load(row + end - 72),
        aie::shuffle_down_fill(dw8_load(row + end - 64), zero, 56));
  } else {
    const dw8_v lo = k == 0 ? aie::shuffle_up_fill(dw8_load(row), zero, 8)
                            : dw8_load(row + k * 64 - 8);
    if constexpr (Chunks == 2)
      return aie::concat(lo,
                         dw8_low(dw8_load(row + k * 64 + 56).extract<32>(0)));
    // Taps 3..7 have zero weights, so what lies past tap 2 does not matter.
    return aie::concat(lo, dw8_load(row + k * 64 + 56));
  }
}

// One channel block, 8 * (Chunks - 1) < input_width <= 8 * Chunks: chunks at
// pixels 0, 8, ... and the last one ending at the row end. keep[0..2] are the
// filter rows' taps (none for a row `check` drops), keep[3] the pixels of a
// single chunk. `last` marks the block that ends an output buffer.
template <int Chunks, bool Aligned>
static inline void
dw8_block(const uint8_t *line0, const uint8_t *line1, const uint8_t *line2,
          const int8_t *wts, const aie::mask<64> *keep, uint8_t *__restrict out,
          const int32_t input_width, const int scale, const bool last) {
  const uint8_t *in[3] = {line0, line1, line2};
  const int32_t end = input_width * 8;
  dw8_w w[3];
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++)
    w[i] = aie::select(
        aie::zeros<int8, 64>(),
        dw8_load((const uint8_t *)(wts + i * 24)).cast_to<int8>(), keep[i]);
  BN3_UNROLL_FULL
  for (int k = 0; k < Chunks; k++) {
    aie::accum<acc32, 64> acc = dw8_conv::mul(
        w[0], 0, dw8_window<Chunks, Aligned>(in[0], k, end, keep[3]), 0);
    BN3_UNROLL_FULL
    for (int i = 1; i < 3; i++)
      acc = dw8_conv::mac(
          acc, w[i], 0, dw8_window<Chunks, Aligned>(in[i], k, end, keep[3]), 0);
    const dw8_v v = dw8_out(acc, scale);
    if constexpr (Chunks == 1) {
      dw8_put<Aligned>(out, v.extract<32>(0));
      dw8_put<Aligned>(
          out + end - 32,
          aie::shuffle_down_fill(v, aie::zeros<uint8, 64>(), end - 32)
              .extract<32>(0),
          last);
    } else if (k == Chunks - 1) {
      dw8_put<Aligned>(out + end - 64, v.extract<32>(0));
      dw8_put<Aligned>(out + end - 32, v.extract<32>(1), last);
    } else {
      dw8_store<Aligned>(out + k * 64, v);
    }
  }
}

// The block loop only overlaps blocks given a minimum trip count.
#define DW8_MIN_BLOCKS 4

template <int Chunks, bool Aligned>
static void
dw8_rows(const uint8_t *__restrict line0, const uint8_t *__restrict line1,
         const uint8_t *__restrict line2, const int8_t *__restrict wts,
         uint8_t *__restrict output1, uint8_t *__restrict output2,
         const int32_t input_width, const int32_t blocks, const int32_t split,
         const aie::mask<64> *keep, const int scale) {
  const int32_t step = input_width * 8;
  uint8_t *__restrict out = output1;
  AIE_LOOP_MIN_ITERATION_COUNT(DW8_MIN_BLOCKS)
  for (int cd = 0; cd < blocks; cd++) {
    dw8_block<Chunks, Aligned>(line0, line1, line2, wts, keep, out, input_width,
                               scale, cd + 1 == split || cd + 1 == blocks);
    line0 += step;
    line1 += step;
    line2 += step;
    wts += 72;
    out = cd + 1 == split ? output2 : out + step;
  }
}

// Rows of 6 to 8 pixels into a 32-byte aligned buffer. A block ends in the
// granule the next one starts in, and an unaligned store reads that granule
// back, which chains the blocks. Instead each block stores the two granules
// from its start, with the previous block's tail in front; what runs past
// them the next block stores. The last block stores as before, after the
// previous one's tail.
static inline __attribute__((always_inline)) void
dw8_stream(const uint8_t *__restrict line0, const uint8_t *__restrict line1,
           const uint8_t *__restrict line2, const int8_t *__restrict wts,
           uint8_t *__restrict out, const int32_t input_width,
           const int32_t blocks, const aie::mask<64> *keep, const int scale) {
  const int32_t end = input_width * 8;
  dw8_v tail = aie::zeros<uint8, 64>();
  int32_t at = 0;
  AIE_LOOP_MIN_ITERATION_COUNT(DW8_MIN_BLOCKS)
  for (int cd = 1; cd < blocks; cd++) {
    const uint8_t *in[3] = {line0, line1, line2};
    aie::accum<acc32, 64> acc;
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      const dw8_w w = aie::select(
          aie::zeros<int8, 64>(),
          dw8_load((const uint8_t *)(wts + i * 24)).cast_to<int8>(), keep[i]);
      const aie::vector<uint8, 128> d =
          dw8_window<1, false>(in[i], 0, end, keep[3]);
      acc = i == 0 ? dw8_conv::mul(w, 0, d, 0) : dw8_conv::mac(acc, w, 0, d, 0);
    }
    const dw8_v v = dw8_out(acc, scale);
    const int32_t r = at & 31;
    uint8_t *__restrict p = out + (at - r);
    const dw8_v head = aie::shuffle_up_fill(v, tail, r);
    aie::store_v(p, head.extract<32>(0));
    aie::store_v(p + 32, head.extract<32>(1));
    tail = aie::shuffle_up_fill(v, v, 64 - end);
    line0 += end;
    line1 += end;
    line2 += end;
    wts += 72;
    at += end;
  }
  aie::store_v(out + (at & ~31),
               aie::shuffle_up_fill(tail, tail, at & 31).extract<32>(0));
  dw8_block<1, false>(line0, line1, line2, wts, keep, out + at, input_width,
                      scale, true);
}

// Stride 1, input_width <= 32 only.
static int32_t dw8_blocks(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                          int8_t *wts, uint8_t *output1, uint8_t *output2,
                          const int32_t input_width, const int32_t channels,
                          const int32_t split, const int32_t stride,
                          const int32_t check, const int scale,
                          const bool aligned) {
  const int32_t blocks = channels / 8;
  if (stride != 1 || DW_WIDTH(input_width) > 32 || blocks < DW8_MIN_BLOCKS)
    return 0;
  const uint64_t taps = (1ull << 24) - 1;
  const aie::mask<64> keep[4] = {
      aie::mask<64>::from_uint64(check == top ? 0 : taps),
      aie::mask<64>::from_uint64(taps),
      aie::mask<64>::from_uint64(check == bottom ? 0 : taps),
      aie::mask<64>::from_uint64(DW_WIDTH(input_width) >= 8
                                     ? ~0ull
                                     : (1ull << (DW_WIDTH(input_width) * 8)) -
                                           1)};
  // Aligned rows only pay off at 25..32 pixels; narrower rows keep one copy.
#define DW8_ROWS(n, a)                                                         \
  dw8_rows<n, a>(line0, line1, line2, wts, output1, output2, input_width,      \
                 blocks, split, keep, scale)
  // Rows of more than 8 pixels read the width at run time: a constant one made
  // their loops slower.
  const int32_t step = input_width * 8;
  const int32_t rest = blocks - split;
  switch ((DW_WIDTH(input_width) + 7) / 8) {
  case 1:
    if (DW_WIDTH(input_width) >= 6 && split > DW8_MIN_BLOCKS &&
        (rest == 0 || rest > DW8_MIN_BLOCKS) &&
        (((uintptr_t)output1 | (uintptr_t)output2) & 31) == 0) {
      for (int h = 0; h < (rest ? 2 : 1); h++) {
        const int32_t at = h ? split * step : 0;
        dw8_stream(line0 + at, line1 + at, line2 + at,
                   wts + (h ? split : 0) * 72, h ? output2 : output1,
                   input_width, h ? rest : split, keep, scale);
      }
    } else {
      DW8_ROWS(1, false);
    }
    break;
  case 2:
    DW8_ROWS(2, false);
    break;
  case 3:
    DW8_ROWS(3, false);
    break;
  default:
    if (aligned)
      DW8_ROWS(4, true);
    else
      DW8_ROWS(4, false);
    break;
  }
#undef DW8_ROWS
  return blocks;
}

// Stride 2 over 64-byte aligned rows of a multiple of 8 pixels. A window of
// two chunks gives the stride-1 sums at pixels 1..8 of the first, whose even
// ones are stride-2 outputs; stored one output later, after the one before.
// Output 0 is the sum at pixel 0.
static inline aie::vector<uint8, 128> dw8_pair(const uint8_t *p) {
  return aie::concat(aie::load_v<64>(p), aie::load_v<64>(p + 64));
}

static void dw8_s2_row(const uint8_t *__restrict line0,
                       const uint8_t *__restrict line1,
                       const uint8_t *__restrict line2, const dw8_w *w,
                       uint8_t *__restrict out, const int32_t output_width,
                       const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  const dw8_v zero = aie::zeros<uint8, 64>();
  aie::accum<acc32, 64> acc;
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    const aie::vector<uint8, 128> d = aie::concat(zero, aie::load_v<64>(in[i]));
    acc = i == 0 ? dw8_conv::mul(w[0], 0, d, 0)
                 : dw8_conv::mac(acc, w[i], 0, d, 0);
  }
  dw8_v prev = dw8_out(acc, scale);
  const int32_t last = output_width * 16 - 64;
  const int32_t pairs = output_width / 8;
  for (int m = 0; m < pairs; m++) {
    // Past the row, `next` only feeds the sum at pixel input_width, not an
    // output. `mid` never clamps, but a load of its own keeps d1 out of d0's
    // registers, without which the loop does not pipeline.
    const int32_t at = m * 128;
    const int32_t mid = at + 64 < last ? at + 64 : last;
    const int32_t next = at + 128 < last ? at + 128 : last;
    aie::accum<acc32, 64> acc0, acc1;
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      const aie::vector<uint8, 128> d0 = dw8_pair(in[i] + at);
      const aie::vector<uint8, 128> d1 = aie::concat(
          aie::load_v<64>(in[i] + mid), aie::load_v<64>(in[i] + next));
      if (i == 0) {
        acc0 = dw8_conv::mul(w[0], 0, d0, 0);
        acc1 = dw8_conv::mul(w[0], 0, d1, 0);
      } else {
        acc0 = dw8_conv::mac(acc0, w[i], 0, d0, 0);
        acc1 = dw8_conv::mac(acc1, w[i], 0, d1, 0);
      }
    }
    const dw8_v cur = aie::filter_odd(
        aie::concat(dw8_out(acc0, scale), dw8_out(acc1, scale)), 8);
    const dw8_v v = aie::shuffle_up_fill(cur, prev, 8);
    aie::store_v(out + m * 64, v.extract<32>(0));
    aie::store_v(out + m * 64 + 32, v.extract<32>(1));
    prev = cur;
  }
  if (output_width & 4) {
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      const aie::vector<uint8, 128> d =
          aie::concat(aie::load_v<64>(in[i] + pairs * 128), zero);
      acc = i == 0 ? dw8_conv::mul(w[0], 0, d, 0)
                   : dw8_conv::mac(acc, w[i], 0, d, 0);
    }
    aie::store_v(out + pairs * 64,
                 aie::shuffle_up_fill(aie::filter_odd(dw8_out(acc, scale), 8),
                                      prev.extract<32>(1), 8));
  }
}

// Stride 1 over the same rows, wider than dw8_blocks takes: each window of
// two chunks gives the sums at pixels 1..8 of the first, as in dw8_s2_row.
static inline void dw8_s1_row(const uint8_t *__restrict line0,
                              const uint8_t *__restrict line1,
                              const uint8_t *__restrict line2, const dw8_w *w,
                              uint8_t *__restrict out,
                              const int32_t input_width, const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  const dw8_v zero = aie::zeros<uint8, 64>();
  aie::accum<acc32, 64> acc;
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    const aie::vector<uint8, 128> d = aie::concat(zero, aie::load_v<64>(in[i]));
    acc = i == 0 ? dw8_conv::mul(w[0], 0, d, 0)
                 : dw8_conv::mac(acc, w[i], 0, d, 0);
  }
  dw8_v prev = dw8_out(acc, scale);
  const int32_t last = input_width * 8 - 64;
  BN3_UNROLL_FULL
  for (int32_t at = 0; at < last; at += 64) {
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++) {
      const aie::vector<uint8, 128> d = dw8_pair(in[i] + at);
      acc = i == 0 ? dw8_conv::mul(w[0], 0, d, 0)
                   : dw8_conv::mac(acc, w[i], 0, d, 0);
    }
    const dw8_v cur = dw8_out(acc, scale);
    const dw8_v v = aie::shuffle_up_fill(cur, prev, 8);
    aie::store_v(out + at, v.extract<32>(0));
    aie::store_v(out + at + 32, v.extract<32>(1));
    prev = cur;
  }
  BN3_UNROLL_FULL
  for (int i = 0; i < 3; i++) {
    const aie::vector<uint8, 128> d =
        aie::concat(aie::load_v<64>(in[i] + last), zero);
    acc = i == 0 ? dw8_conv::mul(w[0], 0, d, 0)
                 : dw8_conv::mac(acc, w[i], 0, d, 0);
  }
  const dw8_v v = aie::shuffle_up_fill(dw8_out(acc, scale), prev, 8);
  aie::store_v(out + last, v.extract<32>(0));
  aie::store_v(out + last + 32, v.extract<32>(1));
}

// Stride 2 when dw8_s2_row applies, and stride 1 over rows wider than
// dw8_blocks takes.
static int32_t dw8_s2_blocks(const uint8_t *line0, const uint8_t *line1,
                             const uint8_t *line2, const int8_t *wts,
                             uint8_t *output1, uint8_t *output2,
                             const int32_t input_width, const int32_t channels,
                             const int32_t split, const int32_t stride,
                             const int32_t check, const int scale) {
  const int32_t output_width = input_width / stride;
  const uint64_t taps = (1ull << 24) - 1;
  const aie::mask<64> keep[3] = {
      aie::mask<64>::from_uint64(check == top ? 0 : taps),
      aie::mask<64>::from_uint64(taps),
      aie::mask<64>::from_uint64(check == bottom ? 0 : taps)};
  const int32_t step = input_width * 8;
  if (stride == 1) {
    const auto block = [&](int cd) __attribute__((always_inline)) {
      dw8_w w[3];
      BN3_UNROLL_FULL
      for (int i = 0; i < 3; i++)
        w[i] = aie::select(
            aie::zeros<int8, 64>(),
            dw8_load((const uint8_t *)(wts + cd * 72 + i * 24)).cast_to<int8>(),
            keep[i]);
      uint8_t *out = cd < split ? output1 + cd * output_width * 8
                                : output2 + (cd - split) * output_width * 8;
      dw8_s1_row(line0 + cd * step, line1 + cd * step, line2 + cd * step, w,
                 out, input_width, scale);
    };
    if (channels / 8 >= DW8_MIN_BLOCKS) {
      AIE_LOOP_MIN_ITERATION_COUNT(DW8_MIN_BLOCKS)
      for (int cd = 0; cd < channels / 8; cd++)
        block(cd);
    } else {
      for (int cd = 0; cd < channels / 8; cd++)
        block(cd);
    }
    return channels / 8;
  }
  for (int cd = 0; cd < channels / 8; cd++) {
    dw8_w w[3];
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++)
      w[i] = aie::select(
          aie::zeros<int8, 64>(),
          dw8_load((const uint8_t *)(wts + cd * 72 + i * 24)).cast_to<int8>(),
          keep[i]);
    uint8_t *out = cd < split ? output1 + cd * output_width * 8
                              : output2 + (cd - split) * output_width * 8;
    dw8_s2_row(line0 + cd * step, line1 + cd * step, line2 + cd * step, w, out,
               output_width, scale);
  }
  return channels / 8;
}

// Stride 2 over rows dw8_s2_row doesn't take, of an even 9 to 32 pixels: the
// outputs are the even pixels of dw8_block's stride-1 sums, so each chunk
// stores four. The last chunk ends at the row end and its outputs overlap
// the chunk's before.
template <int Chunks>
static void dw8_s2_narrow_rows(const uint8_t *line0, const uint8_t *line1,
                               const uint8_t *line2, const int8_t *wts,
                               uint8_t *output1, uint8_t *output2,
                               const int32_t input_width, const int32_t blocks,
                               const int32_t split, const aie::mask<64> *keep,
                               const int scale) {
  const int32_t end = input_width * 8;
  const int32_t out_end = input_width * 4;
  const auto block = [&](int cd) __attribute__((always_inline)) {
    const uint8_t *in[3] = {line0 + cd * end, line1 + cd * end,
                            line2 + cd * end};
    const int8_t *bw = wts + cd * 72;
    dw8_w w[3];
    BN3_UNROLL_FULL
    for (int i = 0; i < 3; i++)
      w[i] = aie::select(
          aie::zeros<int8, 64>(),
          dw8_load((const uint8_t *)(bw + i * 24)).cast_to<int8>(), keep[i]);
    uint8_t *__restrict out =
        cd < split ? output1 + cd * out_end : output2 + (cd - split) * out_end;
    BN3_UNROLL_FULL
    for (int k = 0; k < Chunks; k++) {
      aie::accum<acc32, 64> acc = dw8_conv::mul(
          w[0], 0, dw8_window<Chunks, false>(in[0], k, end, keep[3]), 0);
      BN3_UNROLL_FULL
      for (int i = 1; i < 3; i++)
        acc = dw8_conv::mac(
            acc, w[i], 0, dw8_window<Chunks, false>(in[i], k, end, keep[3]), 0);
      const dw_v v = aie::filter_even(dw8_out(acc, scale), 8);
      if (k == Chunks - 1)
        dw8_put<false>(out + out_end - 32, v,
                       cd + 1 == split || cd + 1 == blocks);
      else
        dw8_put<false>(out + k * 32, v);
    }
  };
  if (blocks >= DW8_MIN_BLOCKS) {
    AIE_LOOP_MIN_ITERATION_COUNT(DW8_MIN_BLOCKS)
    for (int cd = 0; cd < blocks; cd++)
      block(cd);
  } else {
    for (int cd = 0; cd < blocks; cd++)
      block(cd);
  }
}

// Stride 2 rows dw8_s2_blocks does not take, even widths 10 to 32.
static int32_t dw8_s2_narrow(const uint8_t *line0, const uint8_t *line1,
                             const uint8_t *line2, const int8_t *wts,
                             uint8_t *output1, uint8_t *output2,
                             const int32_t input_width, const int32_t channels,
                             const int32_t split, const int32_t check,
                             const int scale) {
  if (input_width <= 8 || input_width > 32 || input_width % 2)
    return 0;
  const uint64_t taps = (1ull << 24) - 1;
  const aie::mask<64> keep[4] = {
      aie::mask<64>::from_uint64(check == top ? 0 : taps),
      aie::mask<64>::from_uint64(taps),
      aie::mask<64>::from_uint64(check == bottom ? 0 : taps),
      aie::mask<64>::from_uint64(~0ull)};
#define DW8_S2_ROWS(n)                                                         \
  dw8_s2_narrow_rows<n>(line0, line1, line2, wts, output1, output2,            \
                        input_width, channels / 8, split, keep, scale)
  switch ((input_width + 7) / 8) {
  case 2:
    DW8_S2_ROWS(2);
    break;
  case 3:
    DW8_S2_ROWS(3);
    break;
  default:
    DW8_S2_ROWS(4);
    break;
  }
#undef DW8_S2_ROWS
  return channels / 8;
}
#endif

#if AIE_TUNED_AIE2P && defined(CONV_INPUT_WIDTH) &&                            \
    defined(CONV_OUTPUT_CHANNELS)
#define DWG_ENABLED 1
#include <utility>

template <int N, typename F>
static inline __attribute__((always_inline)) void dwg_for(F &&f) {
  [&]<int... K>(std::integer_sequence<int, K...>)
      __attribute__((always_inline)) {
        (f(std::integral_constant<int, K>{}), ...);
      }(std::make_integer_sequence<int, N>{});
}

// A group of NB channel blocks whose rows start and end 64-byte aligned,
// taken as one run of NB * W pixels. Chunk k convolves words k and k + 1 into
// pixels 8k+1..8k+8; chunk -1, over a zero word, yields pixel 0. A chunk sums
// once per block its outputs belong to, with the pixels across that block's
// edges zeroed, and takes each lane from its own block. Stride 2 keeps the
// even pixels.
template <int W, int NB, int S>
struct dwg_geo {
  static constexpr int P = NB * W;
  static constexpr int NW = (P + 7) / 8;
  static constexpr int first(int k, int b) {
    const int a = 8 * k + 1 > b * W ? 8 * k + 1 : b * W;
    return S == 2 && a % 2 ? a + 1 : a;
  }
  static constexpr int last(int k, int b) {
    const int z = 8 * k + 8 < b * W + W - 1 ? 8 * k + 8 : b * W + W - 1;
    return S == 2 && z % 2 ? z - 1 : z;
  }
  static constexpr bool has(int k, int b) { return first(k, b) <= last(k, b); }
  static constexpr int bmin(int k) {
    for (int b = 0; b < NB; b++)
      if (has(k, b))
        return b;
    return NB;
  }
  static constexpr int kmin(int b) {
    for (int k = -1; k < NW; k++)
      if (has(k, b))
        return k;
    return NW;
  }
  static constexpr bool loaded(int i) { return i >= 0 && 8 * i < P; }
  static constexpr int accs(int k) {
    int n = 0;
    for (int j = -1; j <= k; j++)
      for (int b = 0; b < NB; b++)
        n += has(j, b);
    return n;
  }
};

template <int Bytes>
static inline __attribute__((always_inline)) void dwg_put(uint8_t *p, dw8_v v) {
  if constexpr (Bytes >= 64) {
    aie::store_v(p, v);
  } else {
    if constexpr (Bytes >= 32)
      aie::store_v(p, v.extract<32>(0));
    if constexpr (Bytes % 32)
      aie::store_v(p + (Bytes & 32), v.extract<16>((Bytes & 32) / 16));
  }
}

template <int W, int NB, int S>
static inline __attribute__((always_inline)) void
dwg_group(const uint8_t *__restrict line0, const uint8_t *__restrict line1,
          const uint8_t *__restrict line2, const int8_t *__restrict wts,
          uint8_t *__restrict out, const aie::mask<64> *keep, const int scale) {
  using G = dwg_geo<W, NB, S>;
  constexpr int P = G::P, NW = G::NW;
  const uint8_t *in[3] = {line0, line1, line2};
  const dw8_v zero = aie::zeros<uint8, 64>();
  const auto word = [&](auto r, auto i) __attribute__((always_inline)) {
    if constexpr (G::loaded(i))
      return aie::load_v<64>(in[r] + 64 * i);
    else
      return zero;
  };
  // Only the first two pixels of a window's upper word reach the sum.
  const auto head = [&](auto r, auto i) __attribute__((always_inline)) {
    if constexpr (G::loaded(i))
      return aie::load_v<32>(in[r] + 64 * i).template grow<64>();
    else
      return zero;
  };
  const auto clear = [&](dw8_v x, int z,
                         int base) __attribute__((always_inline)) {
    return aie::select(x, zero,
                       aie::mask<64>::from_uint64(0xffull << (8 * (z - base))));
  };
  dw8_w w[NB][3];
  dw8_v v[NW + 1];
  dw8_v prev;
  dwg_for<NW + 1>([&](auto kk) __attribute__((always_inline)) {
    constexpr int k = kk - 1;
    dwg_for<NB>([&](auto bb) __attribute__((always_inline)) {
      constexpr int b = bb;
      if constexpr (G::has(k, b)) {
        if constexpr (G::kmin(b) == k) {
          dwg_for<3>([&](auto r) __attribute__((always_inline)) {
            w[b][r] =
                aie::select(aie::zeros<int8, 64>(),
                            dw8_load((const uint8_t *)(wts + b * 72 + r * 24))
                                .cast_to<int8>(),
                            keep[r]);
          });
        }
        constexpr int f = G::first(k, b), bs = b * W, be = bs + W;
        constexpr bool zl = f == bs && bs > 0;
        constexpr bool zr = G::last(k, b) == be - 1 && G::loaded(be / 8);
        static_assert(!zr || be < 8 * k + 8);
        aie::accum<acc32, 64> acc;
        dwg_for<3>([&](auto r) __attribute__((always_inline)) {
          dw8_v lo = word(r, std::integral_constant<int, k>{});
          const dw8_v hi = head(r, std::integral_constant<int, k + 1>{});
          if constexpr (zl)
            lo = clear(lo, bs - 1, 8 * k);
          if constexpr (zr)
            lo = clear(lo, be, 8 * k);
          const aie::vector<uint8, 128> d = aie::concat(lo, hi);
          acc = r == 0 ? dw8_conv::mul(w[b][r], 0, d, 0)
                       : dw8_conv::mac(acc, w[b][r], 0, d, 0);
        });
        const dw8_v o = dw8_out(acc, scale);
        if constexpr (G::bmin(k) == b)
          v[k + 1] = o;
        else
          v[k + 1] = aie::select(
              v[k + 1], o,
              aie::mask<64>::from_uint64(~0ull << (8 * (f - 8 * k - 1))));
      }
    });
    if constexpr (G::accs(k) / 4 != G::accs(k - 1) / 4)
      __builtin_aie2p_sched_barrier();
    if constexpr (G::bmin(k) == NB)
      v[k + 1] = zero;
    if constexpr (S == 1 && k >= 0) {
      dwg_put<8 * P - 64 * k>(out + 64 * k,
                              aie::shuffle_up_fill(v[k + 1], v[k], 8));
    } else if constexpr (S == 2 && k == -1) {
      prev = v[0];
    } else if constexpr (S == 2 && (k % 2 == 1 || k == NW - 1)) {
      constexpr int m = k / 2;
      const dw8_v e = aie::filter_odd(
          aie::concat(v[2 * m + 1], k % 2 ? v[k + 1] : zero), 8);
      dwg_put<4 * P - 64 * m>(out + 64 * m, aie::shuffle_up_fill(e, prev, 8));
      prev = e;
    }
  });
}

template <int W, int NB, int S>
static __attribute__((noinline)) void
dwg_tail(const uint8_t *line0, const uint8_t *line1, const uint8_t *line2,
         const int8_t *wts, uint8_t *out, const aie::mask<64> *keep,
         const int scale) {
  dwg_group<W, NB, S>(line0, line1, line2, wts, out, keep, scale);
}
static constexpr int dwg_sb(int w, int s) {
  for (int n = 1; n <= 64; n++)
    if ((n * w * 8) % 64 == 0 && (n * w * 8 / s) % 64 == 0)
      return n;
  return 0;
}

// Takes the call when the shape and alignment fit; returns whether it did.
template <int W, int C, int S>
static bool dwg_run(const uint8_t *line0, const uint8_t *line1,
                    const uint8_t *line2, const int8_t *wts, uint8_t *output,
                    const int32_t check, const int scale) {
  constexpr int blocks = C / 8, SB = dwg_sb(W, S);
  constexpr int groups = SB ? blocks / SB : 0, rest = SB ? blocks % SB : 0;
  if constexpr (W < 8 || W % 2 || C % 8 || SB < 2 || SB * W > 112 ||
                (rest * W * 8 / S) % 16)
    return false;
  else {
    if ((((uintptr_t)line0 | (uintptr_t)line1 | (uintptr_t)line2 |
          (uintptr_t)output) &
         63) != 0)
      return false;
    event0();
    const aie::saturation_mode saved_sat =
        aie::swap_saturation(aie::saturation_mode::saturate);
    const aie::rounding_mode saved_rnd =
        aie::swap_rounding(aie::rounding_mode::conv_even);
    const uint64_t taps = (1ull << 24) - 1;
    const aie::mask<64> keep[3] = {
        aie::mask<64>::from_uint64(check == top ? 0 : taps),
        aie::mask<64>::from_uint64(taps),
        aie::mask<64>::from_uint64(check == bottom ? 0 : taps)};
    constexpr int in_step = SB * W * 8, out_step = in_step / S;
    constexpr int tail_in = groups * in_step, tail_out = groups * out_step;
    if constexpr (rest > 0)
      dwg_tail<W, rest, S>(line0 + tail_in, line1 + tail_in, line2 + tail_in,
                           wts + groups * SB * 72, output + tail_out, keep,
                           scale);
    for (int g = 0; g < groups; g++)
      dwg_group<W, SB, S>(line0 + g * in_step, line1 + g * in_step,
                          line2 + g * in_step, wts + g * SB * 72,
                          output + g * out_step, keep, scale);
    aie::set_rounding(saved_rnd);
    aie::set_saturation(saved_sat);
    event1();
    return true;
  }
}

// Precomputing pixel 0 lets dws2_body traverse channel boundaries without
// masking each convolution. Pack weights for aligned loads; dws2_patch
// restores those boundary outputs afterwards.
template <int W>
__attribute__((always_inline)) static inline void
dws2_stage(const uint8_t *__restrict line0, const uint8_t *__restrict line1,
           const uint8_t *__restrict line2, const int8_t *__restrict wts,
           const aie::mask<64> *keep, int8_t *__restrict wbuf,
           uint8_t *__restrict pix, const int scale) {
  const uint8_t *in[3] = {line0, line1, line2};
  const dw8_v zero = aie::zeros<uint8, 64>();
  aie::vector<int8, 64> v[9];
  dwg_for<9>([&](auto k) __attribute__((always_inline)) {
    v[k] = aie::load_v<64>(wts + 64 * k);
  });
  dw8_v first = zero;
  dwg_for<8>([&](auto cd) __attribute__((always_inline)) {
    aie::accum<acc32, 64> acc;
    dwg_for<3>([&](auto r) __attribute__((always_inline)) {
      constexpr int o = cd * 72 + r * 24, k = o / 64, s = o % 64;
      aie::vector<int8, 64> row;
      if constexpr (s == 0)
        row = v[k];
      else if constexpr (s + 24 <= 64)
        row = aie::shuffle_down(v[k], s);
      else
        row = aie::shuffle_down_fill(v[k], v[k + 1], s);
      const dw8_w w = aie::select(aie::zeros<int8, 64>(), row, keep[r]);
      aie::store_v(wbuf + (cd * 3 + r) * 64, w);
      const aie::vector<uint8, 128> d =
          aie::concat(zero, aie::load_v<64>(in[r] + cd * W * 8));
      acc = r == 0 ? dw8_conv::mul(w, 0, d, 0) : dw8_conv::mac(acc, w, 0, d, 0);
    });
    first = aie::select(
        first,
        aie::shuffle_up(aie::shuffle_down(dw8_out(acc, scale), 56), 8 * cd),
        aie::mask<64>::from_uint64(0xffull << (8 * cd)));
  });
  aie::store_v(pix, first);
}

template <int W, int C>
__attribute__((always_inline)) static inline void
dws2_body(const uint8_t *__restrict line0, const uint8_t *__restrict line1,
          const uint8_t *__restrict line2, const int8_t *__restrict wbuf,
          uint8_t *__restrict output, const int scale) {
  constexpr int NP = W / 16, NT = C / 8 * NP;
  const uint8_t *in[3] = {line0, line1, line2};
  dw8_v prev = aie::zeros<uint8, 64>();
  const int8_t *wp = wbuf;
  int p = 0;
  AIE_LOOP_MIN_ITERATION_COUNT(NT)
  for (int t = 0; t < NT; t++) {
    const int32_t at = 128 * t;
    const int32_t next = t + 1 < NT ? at + 128 : at + 64;
    aie::accum<acc32, 64> acc0, acc1;
    dwg_for<3>([&](auto r) __attribute__((always_inline)) {
      const dw8_w w = aie::load_v<64>(wp + 64 * r);
      const dw8_v a = aie::load_v<64>(in[r] + at);
      const dw8_v b = aie::load_v<64>(in[r] + at + 64);
      const dw8_v bh = b.template extract<32>(0).template grow<64>();
      const dw8_v ch = aie::load_v<32>(in[r] + next).template grow<64>();
      const aie::vector<uint8, 128> d0 = aie::concat(a, bh);
      const aie::vector<uint8, 128> d1 = aie::concat(b, ch);
      if constexpr (r == 0) {
        acc0 = dw8_conv::mul(w, 0, d0, 0);
        acc1 = dw8_conv::mul(w, 0, d1, 0);
      } else {
        acc0 = dw8_conv::mac(acc0, w, 0, d0, 0);
        acc1 = dw8_conv::mac(acc1, w, 0, d1, 0);
      }
    });
    const dw8_v cur = aie::filter_odd(
        aie::concat(dw8_out(acc0, scale), dw8_out(acc1, scale)), 8);
    aie::store_v(output + 64 * t, aie::shuffle_up_fill(cur, prev, 8));
    prev = cur;
    p++;
    wp = p == NP ? wp + 192 : wp;
    p = p == NP ? 0 : p;
  }
}

template <int W>
__attribute__((always_inline)) static inline void
dws2_patch(const uint8_t *__restrict pix, uint8_t *__restrict output) {
  const dw8_v first = aie::load_v<64>(pix);
  dwg_for<8>([&](auto cd) __attribute__((always_inline)) {
    uint8_t *o = output + cd * W * 4;
    aie::store_v(
        o, aie::select(aie::load_v<32>(o),
                       aie::shuffle_down(first, 8 * cd).template extract<32>(0),
                       aie::mask<32>::from_uint32(0xffu)));
  });
}

template <int W, int C>
static bool dws2_run(const uint8_t *line0, const uint8_t *line1,
                     const uint8_t *line2, const int8_t *wts, uint8_t *output,
                     const int32_t check, const int scale) {
  if constexpr (W % 16 || C % 64)
    return false;
  else {
    if ((((uintptr_t)line0 | (uintptr_t)line1 | (uintptr_t)line2 |
          (uintptr_t)wts | (uintptr_t)output) &
         63) != 0)
      return false;
    event0();
    const aie::saturation_mode saved_sat =
        aie::swap_saturation(aie::saturation_mode::saturate);
    const aie::rounding_mode saved_rnd =
        aie::swap_rounding(aie::rounding_mode::conv_even);
    const uint64_t taps = (1ull << 24) - 1;
    const aie::mask<64> keep[3] = {
        aie::mask<64>::from_uint64(check == top ? 0 : taps),
        aie::mask<64>::from_uint64(taps),
        aie::mask<64>::from_uint64(check == bottom ? 0 : taps)};
    alignas(64) static int8_t wbuf[C / 8 * 3 * 64];
    alignas(64) static uint8_t pix[C];
    for (int g = 0; g < C / 64; g++)
      dws2_stage<W>(line0 + g * W * 64, line1 + g * W * 64, line2 + g * W * 64,
                    wts + g * 576, keep, wbuf + g * 1536, pix + g * 64, scale);
    dws2_body<W, C>(line0, line1, line2, wbuf, output, scale);
    for (int g = 0; g < C / 64; g++)
      dws2_patch<W>(pix + g * 64, output + g * W * 32);
    aie::set_rounding(saved_rnd);
    aie::set_saturation(saved_sat);
    event1();
    return true;
  }
}
#endif

static void dw_vector(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                      int8_t *wts, uint8_t *output1, uint8_t *output2,
                      const int32_t row_width, const int32_t channels,
                      const int32_t split, const int32_t stride,
                      const int32_t check, const int scale) {
  event0();
  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const int32_t input_width = DW_WIDTH(row_width);
  const int32_t output_width = input_width / stride;
#if AIE_TUNED_AIE2P
  // Stride 2 loads 64 bytes at a time, which AIE2P wants 64-byte aligned.
  const uintptr_t align = stride == 1 ? 31 : 63;
#else
  const uintptr_t align = 31;
#endif
  const bool aligned =
      (((uintptr_t)line0 | (uintptr_t)line1 | (uintptr_t)line2 |
        (uintptr_t)output1 | (uintptr_t)output2) &
       align) == 0 &&
      (stride == 1 ? input_width % 4 : input_width % 8 + output_width % 4) == 0;
#if AIE_TUNED_AIE2P
  const bool rows64 = (((uintptr_t)line0 | (uintptr_t)line1 | (uintptr_t)line2 |
                        (uintptr_t)output1 | (uintptr_t)output2) &
                       63) == 0 &&
                      input_width % 8 == 0;
  // Each walker returns how many channel blocks it covered; the loop below
  // takes the rest.
  const int32_t first =
      (stride == 2 && aligned) || (stride == 1 && rows64 && input_width > 32)
          ? dw8_s2_blocks(line0, line1, line2, wts, output1, output2,
                          input_width, channels, split, stride, check, scale)
      : stride == 2
          ? dw8_s2_narrow(line0, line1, line2, wts, output1, output2,
                          input_width, channels, split, check, scale)
          : dw8_blocks(line0, line1, line2, wts, output1, output2,
                       input_width <= 8 ? input_width : row_width, channels,
                       split, stride, check, scale, aligned);
#else
  const int32_t first = 0;
#endif
  dw_w w[DW_W];
  for (int cd = first; cd < channels / 8; cd++) {
    const int32_t in_off = cd * input_width * 8;
    uint8_t *out = cd < split ? output1 + cd * output_width * 8
                              : output2 + (cd - split) * output_width * 8;
    uint8_t *l0 = line0 + in_off, *l1 = line1 + in_off, *l2 = line2 + in_off;
    dw_taps(wts + cd * 72, check, w);
    if (stride == 1) {
      if (aligned)
        dw_s1_row_aligned(l0, l1, l2, w, out, input_width, scale);
      else
        // A constant input_width here takes 4-6% more cycles.
        dw_s1_row(l0, l1, l2, w, out, row_width, scale);
    } else {
      if (aligned)
        dw_s2_row<true>(l0, l1, l2, w, out, output_width, scale);
      else
        dw_s2_row<false>(l0, l1, l2, w, out, output_width, scale);
    }
  }
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}
#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

#ifdef CONV_BIAS
//*****************************************************************************
// depthwise conv2d 3x3 - vector, with a bias and table, see conv2dk1_i8.cc
//
// Weights [C/8][3][3][c8]. The output row for line1, 8 pixels a block where
// the width is a multiple of 8, else 4, each tap a diagonal mmul; the borders
// shift in zeros.
//*****************************************************************************

// NB neighbouring blocks of M pixels from x0, for one group of 8 channels:
// each tap an mmul by its diagonal 8x8 weight matrix, each matrix loaded
// once for all the blocks, each input vector once for the blocks it spans.
template <int M, int NB>
static inline __attribute__((always_inline)) void
dw_i8_bias_blocks(const int8_t *const *rows, const int8_t *__restrict diag,
                  aie::accum<acc32, M * 8> seeded, int8_t *__restrict output,
                  const int x0, const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int V = M * 8;
  using MMUL = aie::mmul<M, 8, 8, int8, int8>;
  const aie::vector<int8, V> zeros = aie::zeros<int8, V>();

  MMUL acc[NB];
  AIE_LOOP_UNROLL_FULL
  for (int b = 0; b < NB; b++)
    acc[b] = seeded;
  AIE_LOOP_UNROLL_FULL
  for (int r = 0; r < 3; r++) {
    const int8_t *row = rows[r];
    aie::vector<int8, 64> w0 = aie::load_v<64>(diag + (r * 3) * 64);
    aie::vector<int8, 64> w1 = aie::load_v<64>(diag + (r * 3 + 1) * 64);
    aie::vector<int8, 64> w2 = aie::load_v<64>(diag + (r * 3 + 2) * 64);
    // in[b] holds pixels x0 + (b - 1) * M onwards.
    aie::vector<int8, V> in[NB + 2];
    in[0] = x0 > 0 ? aie::load_v<V>(row + (x0 - M) * 8) : zeros;
    AIE_LOOP_UNROLL_FULL
    for (int b = 0; b < NB; b++)
      in[b + 1] = aie::load_v<V>(row + (x0 + b * M) * 8);
    in[NB + 1] =
        x0 + NB * M < iw ? aie::load_v<V>(row + (x0 + NB * M) * 8) : zeros;
    AIE_LOOP_UNROLL_FULL
    for (int b = 0; b < NB; b++) {
      acc[b].mac(aie::shuffle_down_fill(in[b], in[b + 1], V - 8), w0);
      acc[b].mac(in[b + 1], w1);
      acc[b].mac(aie::shuffle_down_fill(in[b + 1], in[b + 2], 8), w2);
    }
  }
  AIE_LOOP_UNROLL_FULL
  for (int b = 0; b < NB; b++)
    aie::store_v(output + b * V, acc[b].template to_vector<int8>(scale));
}

template <bool Table>
static void
dw_i8_bias_vector(const int8_t *line0, const int8_t *line1, const int8_t *line2,
                  const int8_t *__restrict wts, const int32_t *__restrict bias,
                  const int8_t *__restrict table, int8_t *__restrict output,
                  const int32_t check, const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int channels = CONV_OUTPUT_CHANNELS;
  // 8 pixels a vector where every row starts 64-byte aligned.
  constexpr int M = iw % 8 == 0 ? 8 : 4;
  constexpr int V = M * 8;
  constexpr int NB = 2;
  constexpr int groups = iw / M / NB;
  constexpr int singles = iw / M % NB;
  static_assert(iw % 4 == 0, "the depthwise bias kernel tiles the row by 4");
  event0();

  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd = aie::swap_rounding(CONV_BIAS_ROUNDING);

  const int row_start = check == top ? 1 : 0;
  const int row_end = check == bottom ? 2 : 3;
  const int8_t *lines[3] = {line0, line1, line2};
  // Lane ic * 8 + oc of a tap's matrix holds that tap's channel-oc weight
  // where ic == oc; a border row's matrices are zero.
  const auto on_diagonal = aie::mask<64>::from_uint64(0x8040201008040201ull);
  const auto odd = aie::mask<16>::from_uint32(0xaaaaaaaau);
  alignas(64) int8_t diag[9 * 64];
  for (int c = 0; c < channels / 8; c++) {
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < 9; t++) {
      const int32_t *tap =
          reinterpret_cast<const int32_t *>(wts + (c * 9 + t) * 8);
      const bool live = t / 3 >= row_start && t / 3 < row_end;
      aie::vector<int8, 64> taps = aie::vector_cast<int8>(
          aie::select(aie::broadcast<int32, 16>(tap[0]),
                      aie::broadcast<int32, 16>(tap[1]), odd));
      aie::store_v(diag + t * 64,
                   live ? aie::select(aie::zeros<int8, 64>(), taps, on_diagonal)
                        : aie::zeros<int8, 64>());
    }
    const aie::accum<acc32, V> seeded =
        bias_seed<M>(aie::load_v<8>(bias + c * 8));
    const int8_t *rows[3] = {lines[0] + c * iw * 8, lines[1] + c * iw * 8,
                             lines[2] + c * iw * 8};
    int8_t *out = output + c * iw * 8;
    for (int g = 0; g < groups; g++)
      dw_i8_bias_blocks<M, NB>(rows, diag, seeded, out + g * NB * V, g * NB * M,
                               scale);
    for (int x = 0; x < singles; x++)
      dw_i8_bias_blocks<M, 1>(rows, diag, seeded, out + (groups * NB + x) * V,
                              (groups * NB + x) * M, scale);
  }
  if constexpr (Table)
    map_table_i8<channels * iw>(output, table);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

#if AIE_TUNED_AIE2P
// Rows of 8k pixels: AIE2P convolves 8 pixels x 8 channels over a filter row
// in one op (see dw8_conv). Chunk k of a row convolves the words k and k + 1
// into output pixels 8k+1..8k+8, seeded with the bias; chunk -1, over a zero
// word, gives pixel 0, and the word past the row is zero. Output word k is
// chunk k - 1's last pixel and chunk k's first seven.
using dwb_conv = aie::sliding_mul_ch_ops<8, 8, 8, 1, 1, 1, int8, int8>;

template <int N, typename F>
static inline __attribute__((always_inline)) void dwb_for(F &&f) {
  [&]<int... K>(std::integer_sequence<int, K...>)
      __attribute__((always_inline)) {
        (f(std::integral_constant<int, K>{}), ...);
      }(std::make_integer_sequence<int, N>{});
}

template <bool Table>
static void
dw_i8_bias_slide(const int8_t *__restrict line0, const int8_t *__restrict line1,
                 const int8_t *__restrict line2, const int8_t *__restrict wts,
                 const int32_t *__restrict bias, const int8_t *__restrict table,
                 int8_t *__restrict output, const int32_t check,
                 const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int channels = CONV_OUTPUT_CHANNELS;
  constexpr int words = iw / 8;
  static_assert(iw % 8 == 0, "the sliding path takes rows of 8k pixels");
  event0();

  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd = aie::swap_rounding(CONV_BIAS_ROUNDING);

  const aie::vector<int8, 64> zero = aie::zeros<int8, 64>();
  // A filter row's three taps are the coefficient's first 24 lanes; a border
  // row's are zero.
  const uint64_t taps = (1ull << 24) - 1;
  const aie::mask<64> keep[3] = {
      aie::mask<64>::from_uint64(check == top ? 0 : taps),
      aie::mask<64>::from_uint64(taps),
      aie::mask<64>::from_uint64(check == bottom ? 0 : taps)};
  for (int c = 0; c < channels / 8; c++) {
    aie::vector<int8, 64> w[3];
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < 3; r++)
      w[r] = aie::select(
          zero, aie::load_unaligned_v<64>(wts + c * 72 + r * 24, 4), keep[r]);
    const aie::accum<acc32, 64> seeded =
        bias_seed<8>(aie::load_v<8>(bias + c * 8));
    const int8_t *__restrict r0 = line0 + c * iw * 8;
    const int8_t *__restrict r1 = line1 + c * iw * 8;
    const int8_t *__restrict r2 = line2 + c * iw * 8;
    int8_t *__restrict out = output + c * iw * 8;

    // Pixels 8k + 1..8k + 8 from the rows' words k (zero at -1) and k + 1
    // (zero past the row); only the first two pixels of word k + 1 reach it.
    auto chunk = [&](auto first, auto last,
                     int k) __attribute__((always_inline)) {
      aie::accum<acc32, 64> acc = seeded;
      AIE_LOOP_UNROLL_FULL
      for (int r = 0; r < 3; r++) {
        const int8_t *__restrict row = r == 0 ? r0 : (r == 1 ? r1 : r2);
        aie::vector<int8, 64> lo, hi;
        if constexpr (decltype(first)::value)
          lo = zero;
        else
          lo = aie::load_v<64>(row + 64 * k);
        if constexpr (decltype(last)::value)
          hi = zero;
        else
          hi = aie::load_v<32>(row + 64 * (k + 1)).template grow<64>();
        acc = dwb_conv::mac(acc, w[r], 0, aie::concat(lo, hi), 0);
      }
      return acc.template to_vector<int8>(scale);
    };
    using T = std::true_type;
    using F = std::false_type;
    aie::vector<int8, 64> prev = chunk(T{}, F{}, -1);
    if constexpr (words <= 5) {
      // A short row is straight-line code, so the channel loop pipelines.
      dwb_for<words - 1>([&](auto kk) __attribute__((always_inline)) {
        constexpr int k = decltype(kk)::value;
        const aie::vector<int8, 64> a = chunk(F{}, F{}, k);
        aie::store_v(out + 64 * k, aie::shuffle_up_fill(a, prev, 8));
        prev = a;
      });
      const aie::vector<int8, 64> last = chunk(F{}, T{}, words - 1);
      aie::store_v(out + 64 * (words - 1), aie::shuffle_up_fill(last, prev, 8));
      continue;
    }
    // Two words a trip, so the loop's latency spreads over twice the work;
    // an odd word before the last goes on its own.
    constexpr int pairs = (words - 1) / 2;
    for (int k = 0; k < 2 * pairs; k += 2) {
      const aie::vector<int8, 64> a = chunk(F{}, F{}, k);
      const aie::vector<int8, 64> b = chunk(F{}, F{}, k + 1);
      aie::store_v(out + 64 * k, aie::shuffle_up_fill(a, prev, 8));
      aie::store_v(out + 64 * (k + 1), aie::shuffle_up_fill(b, a, 8));
      prev = b;
    }
    if constexpr ((words - 1) % 2) {
      const aie::vector<int8, 64> a = chunk(F{}, F{}, words - 2);
      aie::store_v(out + 64 * (words - 2), aie::shuffle_up_fill(a, prev, 8));
      prev = a;
    }
    const aie::vector<int8, 64> cur = chunk(F{}, T{}, words - 1);
    aie::store_v(out + 64 * (words - 1), aie::shuffle_up_fill(cur, prev, 8));
  }
  if constexpr (Table)
    map_table_i8<channels * iw>(output, table);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}
#endif

#if AIE_TUNED_AIE2P
// Rows of 8k + 4 pixels, channel blocks in pairs: two blocks' rows are one
// run of 2W = 8(2q + 1) pixels, every pair 64-byte aligned. Chunk k is as in
// dw_i8_bias_slide over the run, with the first block's weights before chunk
// q and the second's after it. Chunk q, whose outputs 8q+1..8q+8 straddle
// the blocks' edge at pixel W = 8q + 4, runs once per block with the other
// block's edge pixel zeroed, and its lanes are taken from each in turn.

template <bool Table>
static void
dw_i8_bias_pairs(const int8_t *__restrict line0, const int8_t *__restrict line1,
                 const int8_t *__restrict line2, const int8_t *__restrict wts,
                 const int32_t *__restrict bias, const int8_t *__restrict table,
                 int8_t *__restrict output, const int32_t check,
                 const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int channels = CONV_OUTPUT_CHANNELS;
  constexpr int q = iw / 8;
  constexpr int words = 2 * q + 1;
  static_assert(iw % 8 == 4 && channels % 16 == 0,
                "the pair path takes rows of 8k + 4 pixels, blocks in pairs");
  event0();

  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd = aie::swap_rounding(CONV_BIAS_ROUNDING);

  const aie::vector<int8, 64> zero = aie::zeros<int8, 64>();
  const uint64_t taps = (1ull << 24) - 1;
  const aie::mask<64> keep[3] = {
      aie::mask<64>::from_uint64(check == top ? 0 : taps),
      aie::mask<64>::from_uint64(taps),
      aie::mask<64>::from_uint64(check == bottom ? 0 : taps)};
  // Chunk q's window starts at pixel 8q: the second block's first pixel is
  // window pixel 4, the first block's last is window pixel 3.
  const aie::mask<64> not_second = aie::mask<64>::from_uint64(~(0xffull << 32));
  const aie::mask<64> not_first = aie::mask<64>::from_uint64(~(0xffull << 24));
  // Of chunk q's outputs 8q+1.., the first three are the first block's.
  const aie::mask<64> second_lanes = aie::mask<64>::from_uint64(~0ull << 24);
  for (int c = 0; c < channels / 8; c += 2) {
    aie::vector<int8, 64> w[2][3];
    aie::accum<acc32, 64> seeded[2];
    AIE_LOOP_UNROLL_FULL
    for (int b = 0; b < 2; b++) {
      AIE_LOOP_UNROLL_FULL
      for (int r = 0; r < 3; r++)
        w[b][r] = aie::select(
            zero, aie::load_unaligned_v<64>(wts + (c + b) * 72 + r * 24, 4),
            keep[r]);
      seeded[b] = bias_seed<8>(aie::load_v<8>(bias + (c + b) * 8));
    }
    const int8_t *__restrict rows[3] = {line0 + c * iw * 8, line1 + c * iw * 8,
                                        line2 + c * iw * 8};
    int8_t *__restrict out = output + c * iw * 8;

    auto chunk = [&](auto kk, auto bb,
                     aie::mask<64> keep_window) __attribute__((always_inline)) {
      constexpr int k = decltype(kk)::value;
      constexpr int b = decltype(bb)::value;
      aie::accum<acc32, 64> acc = seeded[b];
      AIE_LOOP_UNROLL_FULL
      for (int r = 0; r < 3; r++) {
        aie::vector<int8, 64> lo, hi;
        if constexpr (k < 0)
          lo = zero;
        else
          lo =
              aie::select(zero, aie::load_v<64>(rows[r] + 64 * k), keep_window);
        if constexpr (k + 1 >= words)
          hi = zero;
        else
          hi = aie::load_v<32>(rows[r] + 64 * (k + 1)).template grow<64>();
        acc = dwb_conv::mac(acc, w[b][r], 0, aie::concat(lo, hi), 0);
      }
      return acc.template to_vector<int8>(scale);
    };
    const aie::mask<64> all = aie::mask<64>::from_uint64(~0ull);
    aie::vector<int8, 64> prev = chunk(std::integral_constant<int, -1>{},
                                       std::integral_constant<int, 0>{}, all);
    dwb_for<words>([&](auto kk) __attribute__((always_inline)) {
      constexpr int k = decltype(kk)::value;
      aie::vector<int8, 64> cur;
      if constexpr (k < q) {
        cur = chunk(kk, std::integral_constant<int, 0>{}, all);
      } else if constexpr (k > q) {
        cur = chunk(kk, std::integral_constant<int, 1>{}, all);
      } else {
        const aie::vector<int8, 64> first =
            chunk(kk, std::integral_constant<int, 0>{}, not_second);
        const aie::vector<int8, 64> second =
            chunk(kk, std::integral_constant<int, 1>{}, not_first);
        cur = aie::select(first, second, second_lanes);
      }
      aie::vector<int8, 64> word = aie::shuffle_up_fill(cur, prev, 8);
      aie::store_v(out + 64 * k, word);
      prev = cur;
    });
  }
  if constexpr (Table)
    map_table_i8<channels * iw>(output, table);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

#endif

// The sliding path where the rows are 8k pixels, else blocks of diagonal
// mmuls.
template <bool Table>
static void dw_i8_bias(const int8_t *line0, const int8_t *line1,
                       const int8_t *line2, const int8_t *wts,
                       const int32_t *bias, const int8_t *table, int8_t *output,
                       const int32_t check, const int scale) {
#if AIE_TUNED_AIE2P
  // The sliding paths load and store whole 64-byte words: the contract asks
  // for 64-byte aligned lines and output.
  if constexpr (CONV_INPUT_WIDTH % 8 == 0) {
    dw_i8_bias_slide<Table>(line0, line1, line2, wts, bias, table, output,
                            check, scale);
    return;
  } else if constexpr (CONV_INPUT_WIDTH % 8 == 4 && CONV_INPUT_WIDTH >= 12 &&
                       CONV_OUTPUT_CHANNELS % 16 == 0) {
    dw_i8_bias_pairs<Table>(line0, line1, line2, wts, bias, table, output,
                            check, scale);
    return;
  }
#endif
  dw_i8_bias_vector<Table>(line0, line1, line2, wts, bias, table, output, check,
                           scale);
}

extern "C" {

#ifdef CONV_TABLE
void DW_BIAS_SYMBOL(int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts,
                    int32_t *bias, int8_t *table, int8_t *output,
                    const int32_t input_width, const int32_t input_channels,
                    const int32_t output_channels, const int32_t kernel_width,
                    const int32_t kernel_height, const int32_t check,
                    const int scale, const int channel_offset) {
  dw_i8_bias<true>(line0, line1, line2, wts, bias, table, output, check, scale);
}
#else
void DW_BIAS_SYMBOL(int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts,
                    int32_t *bias, int8_t *output, const int32_t input_width,
                    const int32_t input_channels, const int32_t output_channels,
                    const int32_t kernel_width, const int32_t kernel_height,
                    const int32_t check, const int scale,
                    const int channel_offset) {
  dw_i8_bias<false>(line0, line1, line2, wts, bias, nullptr, output, check,
                    scale);
}
#endif

} // extern "C"

#else // CONV_BIAS

extern "C" {

#ifdef REGULAR
#ifdef SCALAR
#ifdef STRIDE2
void conv2dk3_dw_stride2_relu_ui8_ui8(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t check, const int scale,
    const int channel_offset) {
#ifdef DWG_ENABLED
  if (dwg_run<CONV_INPUT_WIDTH, CONV_OUTPUT_CHANNELS, 2>(
          line0, line1, line2, wts, output, check, scale))
    return;
  if (dws2_run<CONV_INPUT_WIDTH, CONV_OUTPUT_CHANNELS>(line0, line1, line2, wts,
                                                       output, check, scale))
    return;
#endif
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) / 2 >= 4) {
    dw_vector(line0, line1, line2, wts, output, output, input_width,
              DW_CHANNELS(output_channels), DW_CHANNELS(output_channels) / 8, 2,
              check, scale);
    return;
  }
#endif
  conv2dk3_stride2_ui8_scalar(line0, line1, line2, wts, output, input_width,
                              input_channels, output_channels, kernel_width,
                              kernel_height, check, scale, channel_offset);
}
#else
void conv2dk3_dw_stride1_relu_ui8_ui8(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t check, const int scale,
    const int channel_offset) {
#ifdef DWG_ENABLED
  if (dwg_run<CONV_INPUT_WIDTH, CONV_OUTPUT_CHANNELS, 1>(
          line0, line1, line2, wts, output, check, scale))
    return;
#endif
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) >= 5) {
    dw_vector(line0, line1, line2, wts, output, output, input_width,
              DW_CHANNELS(output_channels), DW_CHANNELS(output_channels) / 8, 1,
              check, scale);
    return;
  }
#endif
  conv2dk3_ui8_scalar(line0, line1, line2, wts, output, input_width,
                      input_channels, output_channels, kernel_width,
                      kernel_height, check, scale, channel_offset);
}
#endif
#endif
#endif

#ifdef BN13
#ifdef SCALAR
#ifdef STRIDE1_OUT_SPLIT
void bn13_conv2dk3_ui8_out_split(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output1, uint8_t *output2, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t check, const int scale, const int channel_offset) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) >= 5) {
    dw_vector(line0, line1, line2, wts, output1, output2, input_width,
              output_channels, output_channels / 16, 1, check, scale);
    return;
  }
#endif
  conv2dk3_ui8_out_split_scalar(line0, line1, line2, wts, output1, output2,
                                input_width, input_channels, output_channels,
                                kernel_width, kernel_height, check, scale,
                                channel_offset);
}
#endif

#ifdef STRIDE1
void bn13_conv2dk3_ui8(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                       int8_t *wts, uint8_t *output, const int32_t input_width,
                       const int32_t input_channels,
                       const int32_t output_channels,
                       const int32_t kernel_width, const int32_t kernel_height,
                       const int32_t check, const int scale,
                       const int channel_offset) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) >= 5) {
    dw_vector(line0, line1, line2, wts, output, output, input_width,
              output_channels, output_channels / 8, 1, check, scale);
    return;
  }
#endif
  conv2dk3_ui8_scalar(line0, line1, line2, wts, output, input_width,
                      input_channels, output_channels, kernel_width,
                      kernel_height, check, scale, channel_offset);
}
#endif
#endif
#endif // BN

#ifdef BN14
#ifdef SCALAR
#ifdef STRIDE1_OUT_SPLIT
void bn14_conv2dk3_ui8_out_split(
    uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
    uint8_t *output1, uint8_t *output2, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t check, const int scale, const int channel_offset) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) >= 5) {
    dw_vector(line0, line1, line2, wts, output1, output2, input_width,
              output_channels, output_channels / 16, 1, check, scale);
    return;
  }
#endif
  conv2dk3_ui8_out_split_scalar(line0, line1, line2, wts, output1, output2,
                                input_width, input_channels, output_channels,
                                kernel_width, kernel_height, check, scale,
                                channel_offset);
}
#endif

#ifdef STRIDE1
void bn14_conv2dk3_ui8(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                       int8_t *wts, uint8_t *output, const int32_t input_width,
                       const int32_t input_channels,
                       const int32_t output_channels,
                       const int32_t kernel_width, const int32_t kernel_height,
                       const int32_t check, const int scale,
                       const int channel_offset) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) >= 5) {
    dw_vector(line0, line1, line2, wts, output, output, input_width,
              output_channels, output_channels / 8, 1, check, scale);
    return;
  }
#endif
  conv2dk3_ui8_scalar(line0, line1, line2, wts, output, input_width,
                      input_channels, output_channels, kernel_width,
                      kernel_height, check, scale, channel_offset);
}
#endif
#endif
#endif // BN
}

#endif // CONV_BIAS
