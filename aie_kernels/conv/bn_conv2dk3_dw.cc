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

// On AIE2P the factories pass the row width, which keeps only the code that
// width takes; the width argument is then not read.
#if AIE_TUNED_AIE2P && defined(CONV_INPUT_WIDTH)
#define DW_WIDTH(w) CONV_INPUT_WIDTH
#else
#define DW_WIDTH(w) (w)
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
// AIE2P keeps these short loops rolled and their register arrays on the
// stack unless told to unroll them; the AIE2 build is left as tuned.
#if AIE_TUNED_AIE2P
#define BN3_UNROLL_FULL AIE_LOOP_UNROLL_FULL
#else
#define BN3_UNROLL_FULL
#endif

// One [W][8] row of a channel block per call; 32 lanes are 4 pixels x 8
// channels. A row dropped by `check` gets zero weights, so its line contents
// never reach the sum.
using dw_v = aie::vector<uint8, 32>;
using dw_w = aie::vector<int8, 32>;

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
      w[r * 3 + ki] = aie::vector_cast<int8>(pair);
    }
  }
}

static inline aie::accum<acc32, 32> dw_sum(const dw_v *l, const dw_v *c,
                                           const dw_v *r, const dw_w *w) {
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
// chunk at input_width - 4 that may overlap the one before it.
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
// differently. Every load here is 512 bits, so a load only covers another at
// the same address.
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
// single chunk.
template <int Chunks, bool Aligned>
static inline void dw8_block(const uint8_t *line0, const uint8_t *line1,
                             const uint8_t *line2, const int8_t *wts,
                             const aie::mask<64> *keep, uint8_t *__restrict out,
                             const int32_t input_width, const int scale) {
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
      dw8_put<Aligned>(out + end - 32, aie::shuffle_down_fill(
                                           v, aie::zeros<uint8, 64>(), end - 32)
                                           .extract<32>(0));
    } else {
      dw8_store<Aligned>(k == Chunks - 1 ? out + end - 64 : out + k * 64, v);
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
                               scale);
    line0 += step;
    line1 += step;
    line2 += step;
    wts += 72;
    out = cd + 1 == split ? output2 : out + step;
  }
}

// Stride 1, input_width <= 32 only; returns how many channel blocks it
// covered.
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
  switch ((DW_WIDTH(input_width) + 7) / 8) {
  case 1:
    DW8_ROWS(1, false);
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
#endif

static void dw_vector(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                      int8_t *wts, uint8_t *output1, uint8_t *output2,
                      const int32_t row_width, const int32_t channels,
                      const int32_t split, const int32_t stride,
                      const int32_t check, const int scale) {
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
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
  const int32_t first =
      dw8_blocks(line0, line1, line2, wts, output1, output2,
                 input_width <= 8 ? input_width : row_width, channels, split,
                 stride, check, scale, aligned);
#else
  const int32_t first = 0;
#endif
  dw_w w[9];
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
        dw_s1_row(l0, l1, l2, w, out, input_width, scale);
    } else {
      if (aligned)
        dw_s2_row<true>(l0, l1, l2, w, out, output_width, scale);
      else
        dw_s2_row<false>(l0, l1, l2, w, out, output_width, scale);
    }
  }
  event1();
}
#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

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
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (DW_WIDTH(input_width) / 2 >= 4) {
    dw_vector(line0, line1, line2, wts, output, output, input_width,
              output_channels, output_channels / 8, 2, check, scale);
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