//===- conv2dk1_skip_init.cc -------------------------------------------------*-
// C++
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

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_arch.h"
#include <aie_api/aie.hpp>

#if AIE_TUNED_AIE2P
#include "bn_conv2dk1_aie2.h"
#endif

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;

// #define INT8_MAX 127
// #define INT8_MIN -128

#if defined(BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH_NEW) ||         \
    defined(BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH_NEW)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, int32_t output_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {
  event0();
#if AIE_TUNED_AIE2P
  if (k1_wts_aligned(kernels)) {
    const int32_t blocks = input_channels / input_split / 8;
    const int32_t row = input_width * 8;
    const int32_t oc_out =
        oc + output_channels / (8 * output_split) * weight_index;
    const aie::vector<int8, 32> ones = aie::broadcast<int8, 32>(1);
    k1_cas_get(
        input, kernels + oc * blocks * 64, output + oc_out * row, row, blocks,
        [=](auto &acc, const int8_t *s) {
          aie::accum<acc32, 32> t = aie::mul(k1_load<false>(s), ones);
          t = aie::mac(t, acc.template to_vector<int8>(scale), ones);
          return t.template to_vector<int8>(skip_scale);
        },
        skip + oc_out * row);
    event1();
    return;
  }
#endif
  int ic, ic8, oc8;
  const int skip_scaleT = skip_scale;

  int pixel_limit = 7;

  // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

  // Determine the start and end of the loop based on the chunk index for
  // weights
  int input_channel_chunk_size = input_channels / input_split;
  int start_ic = 0 * input_channel_chunk_size;
  int end_ic = start_ic + input_channel_chunk_size;
  int pixel = 0;
  int oc_offset = 0;
  int oc8_iter = output_channels / (8 * output_split);

  v16acc64 acc_cas = undef_v16acc64();
  v16int32 v16vec_partial[8] = {};
  v16int32 v16vec_cas[8] = {};

  // Process each pixel across all output channels
  for (pixel = 0; pixel < pixel_limit; pixel++) {
    // Loop over output channels (oc8)
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int last_sum = 0;
      int final_sum = 0;

      // Loop over input channels in chunks of 8
      for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          // int k_base = (0 * (input_channel_chunk_size / 8) * 64) + ((ic -
          // start_ic / 8) * 64) + (ic8 * 8);
          int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
          int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                          ((ic) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }

      sum = current_sum;
      v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
    }

    v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int sum_srs = 0;
      int cascade_sum = 0;
      int skip_temp = 0;
      int32_t skip_sum = 0;
      int skip_sum_srs_final = 0;
      int skip_sum_srs_final_out = 0;

      sum = ext_elem(v16vec_partial[pixel], oc8);
      cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
      // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
      sum_srs = (((sum + cascade_sum) + (1 << (scale - 1)) - 1 +
                  (((sum + cascade_sum) >> scale) & 1)) >>
                 scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < -MAX) ? -MIN : sum_srs;
      oc_offset =
          oc + oc8_iter * (weight_index); // works fine when oc8_iter is 4
      skip_temp = skip[(oc_offset * input_width * 8) + (pixel * 8) + oc8];
      skip_sum = sum_srs + skip_temp;

      if (skip_scaleT > 0)
        skip_sum_srs_final = ((skip_sum + (1 << (skip_scaleT - 1)) - 1 +
                               ((skip_sum >> skip_scaleT) & 1)) >>
                              skip_scaleT);
      else
        skip_sum_srs_final = skip_sum;

      // skip_sum_srs_final = (((skip_sum) + (1 << (skip_scaleT - 1)) - 1 +
      // (((skip_sum) >> skip_scaleT) & 1)) >> skip_scaleT);
      skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                               : (skip_sum_srs_final < -MAX)
                                   ? -MIN
                                   : skip_sum_srs_final; // clip

      output[(oc_offset * input_width * 8) + (pixel * 8) + oc8] =
          skip_sum_srs_final_out;
    }
  }

  event1();
}
#endif

#if defined(BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH) ||             \
    defined(BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {
  event0();
  int ic, ic8, oc8;
  const int skip_scaleT = skip_scale;
  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;
  int pixel_limit = 7;

  // Array of pointers to the accumulators
  v16acc64 *accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
      &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
      &v16acc_partial6, &v16acc_partial7, &v16acc_partial8};

  // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

  // Determine the start and end of the loop based on the chunk index for
  // weights
  int input_channel_chunk_size = input_channels / input_split;
  int start_ic = weight_index * input_channel_chunk_size;
  int end_ic = start_ic + input_channel_chunk_size;
  int pixel = 0;
  // Preload vector register with partial sums from previous iteration
  // v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(),
  // undef_v16int32(), undef_v16int32(),
  //                            undef_v16int32(), undef_v16int32(),
  //                            undef_v16int32(), undef_v16int32()};
  // v16int32 v16vec_cas[8] = {undef_v16int32(), undef_v16int32(),
  // undef_v16int32(), undef_v16int32(),
  //                            undef_v16int32(), undef_v16int32(),
  //                            undef_v16int32(), undef_v16int32()};
  v16acc64 acc_cas = undef_v16acc64();
  v16int32 v16vec_partial[8] = {};
  v16int32 v16vec_cas[8] = {};

  if (weight_index !=
      0) { // Preload vector register with partial sum from previous iteration.
           // If weight is only 1 then we don't have partial sum anyway
    for (pixel = 0; pixel < pixel_limit; pixel++) {
      int x = x_start + pixel;
      if (x < input_width) {
        v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
      }
    }
  }

  // Process each pixel across all output channels
  for (pixel = 0; pixel < pixel_limit; pixel++) {
    // Loop over output channels (oc8)
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int last_sum = 0;
      int final_sum = 0;

      // Loop over input channels in chunks of 8
      for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          // int k_base = (0 * (input_channel_chunk_size / 8) * 64) + ((ic -
          // start_ic / 8) * 64) + (ic8 * 8);
          int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
          int k = kernels[(0 * (input_channel_chunk_size / 8) * 64) +
                          ((ic) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }
      // Extract the partial sum if applicable
      if (weight_index != 0) {
        last_sum = ext_elem(v16vec_partial[pixel], oc8);
      }
      // Transfer scalar sum to vector
      sum = current_sum + last_sum;
      v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
      if (weight_index != (input_split / 2 - 1)) {
        *accumulators[pixel] = lups(v16vec_partial[pixel], 0);
      }
    }

    if (weight_index == (input_split / 2 - 1)) {
      // acc_cas=get_scd_v16acc64();
      // int scale_new=8;
      v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;
        int cascade_sum = 0;
        int skip_temp = 0;
        int32_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        sum = ext_elem(v16vec_partial[pixel], oc8);
        cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
        // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
        sum_srs = (((sum + cascade_sum) + (1 << (scale - 1)) - 1 +
                    (((sum + cascade_sum) >> scale) & 1)) >>
                   scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < -MAX) ? -MIN : sum_srs;

        skip_temp = skip[(oc * input_width * 8) + (pixel * 8) + oc8];
        skip_sum = sum_srs + skip_temp;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        // skip_sum_srs_final = (((skip_sum) + (1 << (skip_scaleT - 1)) - 1 +
        // (((skip_sum) >> skip_scaleT) & 1)) >> skip_scaleT);
        skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                 : (skip_sum_srs_final < -MAX)
                                     ? -MIN
                                     : skip_sum_srs_final; // clip

        output[(oc * input_width * 8) + (pixel * 8) + oc8] =
            skip_sum_srs_final_out;
      }
    }
  }

  event1();
}
#endif

#if defined(BN13_2_PARTIAL_GET_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(
    uint8_t *input, int8_t *kernels, uint8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) {
  event0();
  int ic, ic8, oc8;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;
  int pixel_limit = 7;

  // Array of pointers to the accumulators
  v16acc64 *accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
      &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
      &v16acc_partial6, &v16acc_partial7, &v16acc_partial8};

  // Determine the start and end of the loop based on the chunk index for
  // weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;

  // Use an array to hold partial sums for 8 pixels
  v16int32 v16vec_partial[8] = {};
  v16int32 v16vec_cas[8] = {};

  for (oc8 = 0; oc8 < 8; oc8++) {
    int sum[8] = {0};
    int current_sum[8] = {0};
    int sum_srs[8] = {0};
    int last_sum[8] = {0};
    int cascade_sum = 0;
    int32_t skip_sum = 0;
    int skip_sum_srs_final = 0;
    int skip_sum_srs_final_out = 0;
    int skip_temp = 0;

    if (oc8 == 0 &&
        end_ic == input_channels) { // if final set of input channels, scale the
                                    // final output
      // Get cascade sum
      for (int pixel = 0; pixel < pixel_limit; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
        }
      }
    }
    // Current iteration: go over all the input channels
    for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
      for (ic8 = 0; ic8 < 8; ic8++) {
        for (int pixel = 0; pixel < pixel_limit; pixel++) {
          int x = x_start + pixel;
          if (x < input_width) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(0 * (input_channel_chunk_size / 8) * 64) +
                            ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
            current_sum[pixel] += val * k;
          }
        }
      }
    }
    if (weight_index != 1 && oc8 == 0) { // Preload vector register with partial
                                         // sum from previous iteration
      for (int pixel = 0; pixel < pixel_limit; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
        }
      }
    }

    if (weight_index != 1) { // Extract the partial sum
      for (int pixel = 0; pixel < pixel_limit; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          last_sum[pixel] = ext_elem(v16vec_partial[pixel], oc8);
        }
      }
    }

    // Transfer scalar sum to vector
    for (int pixel = 0; pixel < pixel_limit; pixel++) {
      int x = x_start + pixel;
      if (x < input_width) {
        sum[pixel] = current_sum[pixel] + last_sum[pixel];
        v16vec_partial[pixel] =
            upd_elem(v16vec_partial[pixel], oc8, sum[pixel]);
      }
    }

    if (end_ic == input_channels) { // if final set of input channels, scale the
                                    // final output
      for (int pixel = 0; pixel < pixel_limit; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
          // sum_srs[pixel] = ((cascade_sum) + (1 << (scale - 1))) >> scale;
          sum_srs[pixel] =
              ((sum[pixel] + cascade_sum) + (1 << (scale - 1))) >> scale;
          sum_srs[pixel] = (sum_srs[pixel] > MAX)    ? MAX
                           : (sum_srs[pixel] < -MAX) ? -MIN
                                                     : sum_srs[pixel];

          skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
          skip_sum = sum_srs[pixel] + skip_temp;

          skip_sum_srs_final =
              (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
          skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                   : (skip_sum_srs_final < -MAX)
                                       ? -MIN
                                       : skip_sum_srs_final; // clip
          output[(oc * input_width * 8) + (x * 8) + oc8] =
              skip_sum_srs_final_out;
          // output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs[pixel];
        }
      }
    }

    if (oc8 == 7) { // end of vectorization
      for (int pixel = 0; pixel < pixel_limit; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          *accumulators[pixel] = lups(v16vec_partial[pixel], 0);
        }
      }
    }
  }

  event1();
}

#endif

#ifdef PUT
void conv2dk1_skip_ui8_i8_scalar_cascade_put(uint8_t *input0, int8_t *kernels,
                                             const int32_t input_width,
                                             const int32_t input_channels,
                                             const int32_t output_channels) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;

  // Calculate half the input channels
  const int half_input_channels = input_channels / 2;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        for (ic = 0; ic < half_input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (half_input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];

            sum += val * k;
          }
        }

        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        // sum_srs = (sum_srs > MAX)    ? MAX
        //           : (sum_srs < -MIN) ? -MIN
        //                              : sum_srs; // clip
        v16vec_partial = upd_elem(v16vec_partial, value_index, sum);
        value_index++;
        if (value_index == MAX_VALUES) {
          // Transfer the values from vec to acc
          v16acc_partial = lups(v16vec_partial, 0);
          put_mcd(v16acc_partial); // push over cascade
          // Reset the index
          value_index = 0;
        }
      }
    }
  }

  event1();
}
#endif

#ifdef GET
void conv2dk1_skip_ui8_i8_i8_scalar_cascade_get(
    uint8_t *input0, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  const int half_input_channels = input_channels / 2;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int32_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;

        // Extract cascade sum values when starting a new block
        if (value_index == 0) {
          v16acc_partial = get_scd_v16acc64(); // Get the accumulated values
          v16vec_partial =
              lsrs(v16acc_partial, 0, 0); // Convert accumulator to vector
        }

        // Extract the specific cascade sum for the current index
        int partial_sum = ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = half_input_channels / 8; ic < input_channels / 8; ic++) {

          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (half_input_channels / 8) * 64) +
                            ((ic - half_input_channels / 8) * 64) + (ic8 * 8) +
                            oc8];

            sum += val * k;
          }
        }

        if (value_index == MAX_VALUES) {
          value_index = 0;
        }
        // scale for convolution

        sum = sum + partial_sum;
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        // clip

        // skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        // skip_sum = sum_srs + skip_temp;

        // skip_sum_srs_final =
        //     (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        // skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
        //                          : (skip_sum_srs_final < -MIN)
        //                              ? -MIN
        //                              : skip_sum_srs_final; // clip

        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif

#if defined(REGULAR) || defined(BN0) || defined(BN2) || defined(BN4) ||        \
    defined(BN5) || defined(BN7) || defined(BN8) || defined(BN9) ||            \
    defined(BN11)
#ifdef UNSIGNED_SKIP
void conv2dk1_skip_ui8_ui8_i8_scalar(uint8_t *input0, int8_t *kernels,
                                     int8_t *output, uint8_t *skip,
                                     const int32_t input_width,
                                     const int32_t input_channels,
                                     const int32_t output_channels,
                                     const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int32_t sum = 0;
        int32_t sum_srs = 0;
        int32_t skip_sum = 0;
        int8_t sum_srs_out = 0;
        int32_t skip_sum_srs_final = 0;
        int32_t skip_sum_srs_final_out = 0;
        uint8_t skip_temp = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // scale for convolution
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs =
            ((sum + (1 << (scaleT - 1)) - 1 + ((sum >> scaleT) & 1)) >> scaleT);
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs_out = (sum_srs > INT8_MAX)   ? INT8_MAX
                      : (sum_srs < INT8_MIN) ? INT8_MIN
                                             : sum_srs;
        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs_out + skip_temp;

        if (skip_scaleT > 0)
          skip_sum_srs_final = ((skip_sum + (1 << (skip_scaleT - 1)) - 1 +
                                 ((skip_sum >> skip_scaleT) & 1)) >>
                                skip_scaleT);
        else
          skip_sum_srs_final = skip_sum;
        // skip_sum_srs_final = (skip_sum + (1 << (skip_scaleT - 1))) >>
        // skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > INT8_MAX) ? INT8_MAX
                                 : (skip_sum_srs_final < INT8_MIN)
                                     ? INT8_MIN
                                     : skip_sum_srs_final;

        // output[oc * input_width + x] = skip_sum_srs_final_out;
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

        // output[oc * input_width + x] = sum;
        // output[oc * input_width + x] = sum+skip[oc * input_width + x];
      }
    }
  }

  event1();
}

#else
//*****************************************************************************
// conv2d 1x1 skip - scalar
// act: uint8, wts: int8, skip: int8, out: int8
//*****************************************************************************
static void conv2dk1_skip_ui8_i8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int32_t sum = 0;
        int32_t sum_srs = 0;
        int32_t skip_sum = 0;
        int8_t sum_srs_out = 0;
        int8_t skip_temp = 0;
        int32_t skip_sum_srs_final = 0;
        int8_t skip_sum_srs_final_out = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // scale for convolution
        sum_srs =
            ((sum + (1 << (scaleT - 1)) - 1 + ((sum >> scaleT) & 1)) >> scaleT);
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs_out = (sum_srs > INT8_MAX)   ? INT8_MAX
                      : (sum_srs < INT8_MIN) ? INT8_MIN
                                             : sum_srs;

        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs_out + skip_temp;

        if (skip_scaleT > 0)
          skip_sum_srs_final = ((skip_sum + (1 << (skip_scaleT - 1)) - 1 +
                                 ((skip_sum >> skip_scaleT) & 1)) >>
                                skip_scaleT);
        else
          skip_sum_srs_final = skip_sum;
        skip_sum_srs_final_out = (skip_sum_srs_final > INT8_MAX) ? INT8_MAX
                                 : (skip_sum_srs_final < INT8_MIN)
                                     ? INT8_MIN
                                     : skip_sum_srs_final;

        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;
      }
    }
  }

  event1();
}

#endif
#endif //
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
#include "bn_conv2dk1_aie2.h"

// See k1_chunks in bn_conv2dk1_aie2.h; skip is offset like in and out. The
// requantized conv and the skip are added in 32-bit lanes and requantized by
// skip_scale.
template <bool Aligned, int P, int N, typename TS>
static inline void
k1_skip_chunks(const uint8_t *__restrict in, const int8_t *__restrict wts,
               const TS *__restrict skip, int8_t *__restrict out,
               const int32_t row, const int32_t ic_blocks,
               const int32_t last_off, const int scale, const int skip_scale) {
  constexpr int E = 8 * P;
  using MMUL = aie::mmul<P, 8, 8, uint8, int8>;
  MMUL acc[N];
  aie::vector<int8, 64> b = aie::load_v<64>(wts);
  K1_UNROLL_CHUNKS
  for (int j = 0; j < N; j++)
    acc[j].mul(k1_load<Aligned, E>(in + (j == N - 1 ? last_off : E * j)), b);
#pragma clang loop min_iteration_count(1)
  for (int ic = 1; ic < ic_blocks; ic++) {
    in += row;
    wts += 64;
    b = aie::load_v<64>(wts);
    K1_UNROLL_CHUNKS
    for (int j = 0; j < N; j++)
      acc[j].mac(k1_load<Aligned, E>(in + (j == N - 1 ? last_off : E * j)), b);
  }
  const aie::vector<int8, E> ones = aie::broadcast<int8, E>(1);
  K1_UNROLL_CHUNKS
  for (int j = 0; j < N; j++) {
    const int32_t o = j == N - 1 ? last_off : E * j;
    aie::accum<acc32, E> t = aie::mul(k1_load<Aligned, E>(skip + o), ones);
    t = aie::mac(t, acc[j].template to_vector<int8>(scale), ones);
    k1_store<Aligned>(out + o, t.template to_vector<int8>(skip_scale));
  }
}

template <bool Aligned, int P, typename TS>
static void
k1_skip_rows(const uint8_t *input, const int8_t *kernels, const TS *skip,
             int8_t *output, const int32_t input_width,
             const int32_t input_channels, const int32_t output_channels,
             const int scale, const int skip_scale) {
  constexpr int N = 4;
  constexpr int E = 8 * P;
  const int32_t row = input_width * 8;
  const int32_t ic_blocks = input_channels / 8;
  const int32_t chunks = (input_width + P - 1) / P;
  const int32_t groups = chunks / N;
  const int32_t rem = chunks % N;
  const int32_t tail = (input_width - P) * 8;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *wts = kernels + oc * ic_blocks * 64;
    const TS *s = skip + oc * row;
    int8_t *out = output + oc * row;
    for (int g = 0; g < groups; g++) {
      const int32_t x = g * N * E;
      const int32_t last =
          (rem == 0 && g == groups - 1) ? tail - x : E * (N - 1);
      k1_skip_chunks<Aligned, P, N>(input + x, wts, s + x, out + x, row,
                                    ic_blocks, last, scale, skip_scale);
    }
    const int32_t x = groups * N * E;
    switch (rem) {
    case 1:
      k1_skip_chunks<Aligned, P, 1>(input + x, wts, s + x, out + x, row,
                                    ic_blocks, tail - x, scale, skip_scale);
      break;
    case 2:
      k1_skip_chunks<Aligned, P, 2>(input + x, wts, s + x, out + x, row,
                                    ic_blocks, tail - x, scale, skip_scale);
      break;
    case 3:
      k1_skip_chunks<Aligned, P, 3>(input + x, wts, s + x, out + x, row,
                                    ic_blocks, tail - x, scale, skip_scale);
      break;
    }
  }
}

template <int P, typename TS>
static void
k1_skip_chunked(const uint8_t *input, const int8_t *kernels, int8_t *output,
                const TS *skip, const int32_t input_width,
                const int32_t input_channels, const int32_t output_channels,
                const int scale, const int skip_scale) {
  if (input_width % P == 0 &&
      (((uintptr_t)input | (uintptr_t)output | (uintptr_t)skip) &
       (8 * P - 1)) == 0)
    k1_skip_rows<true, P>(input, kernels, skip, output, input_width,
                          input_channels, output_channels, scale, skip_scale);
  else
    k1_skip_rows<false, P>(input, kernels, skip, output, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

template <typename TS>
static void
k1_skip_vector(const uint8_t *input, const int8_t *kernels, int8_t *output,
               const TS *skip, const int32_t input_width,
               const int32_t input_channels, const int32_t output_channels,
               const int scale, const int skip_scale) {
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
#if defined(K1_WIDTH)
  k1_skip_rows<K1_ALIGNED, K1_P>(input, kernels, skip, output, K1_WIDTH,
                                 input_channels, output_channels, scale,
                                 skip_scale);
#elif AIE_TUNED_AIE2P
  k1_skip_rows<true, 4>(input, kernels, skip, output, input_width,
                        input_channels, output_channels, scale, skip_scale);
#else
  k1_skip_chunked<4>(input, kernels, output, skip, input_width, input_channels,
                     output_channels, scale, skip_scale);
#endif
  event1();
}
#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH_NEW

void bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, int32_t output_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(
      input, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale, input_split, output_split,
      weight_index, x_start, oc);
}
#endif
#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH_NEW

void bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, int32_t output_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(
      input, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale, input_split, output_split,
      weight_index, x_start, oc);
}
#endif
// ///////////////////
#ifdef BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH

void bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(
      input, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale, input_split, weight_index, x_start,
      oc);
}
#endif
#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH

void bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(
    uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    const int32_t input_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(
      input, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale, input_split, weight_index, x_start,
      oc);
}
#endif

#ifdef BN13_2_PARTIAL_GET_I8_CAS_WIDTH
void bn13_2_conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(
    uint8_t *input, int8_t *kernels, uint8_t *output, int8_t *skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int skip_scale,
    int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) {
  conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(
      input, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale, input_split, weight_index, x_start,
      oc);
}
#endif

#ifdef PUT
void conv2dk1_skip_ui8_i8_put(uint8_t *input0, int8_t *kernels,
                              const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels) {
  conv2dk1_skip_ui8_i8_scalar_cascade_put(input0, kernels, input_width,
                                          input_channels, output_channels);
}
#endif // PUT

#ifdef GET

void conv2dk1_skip_ui8_i8_i8_get(uint8_t *input0, int8_t *kernels,
                                 int8_t *output, int8_t *skip,
                                 const int32_t input_width,
                                 const int32_t input_channels,
                                 const int32_t output_channels, const int scale,
                                 const int skip_scale) {
  conv2dk1_skip_ui8_i8_i8_scalar_cascade_get(
      input0, kernels, output, skip, input_width, input_channels,
      output_channels, scale, skip_scale);
}

#endif // GET

#ifdef REGULAR
#ifdef SCALAR

#ifdef UNSIGNED_SKIP

void conv2dk1_skip_ui8_ui8_i8(uint8_t *input0, int8_t *kernels, int8_t *output,
                              uint8_t *skip, const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels, const int scale,
                              const int skip_scale) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (input_width >= 4 && skip_scale >= 0 &&
      k1_fits(input_width, kernels, input0, output, skip)) {
    k1_skip_vector(input0, kernels, output, skip, input_width, input_channels,
                   output_channels, scale, skip_scale);
    return;
  }
#endif
  conv2dk1_skip_ui8_ui8_i8_scalar(input0, kernels, output, skip, input_width,
                                  input_channels, output_channels, scale,
                                  skip_scale);
}

#else

void conv2dk1_skip_ui8_i8_i8(uint8_t *input0, int8_t *kernels, int8_t *output,
                             int8_t *skip, const int32_t input_width,
                             const int32_t input_channels,
                             const int32_t output_channels, const int scale,
                             const int skip_scale) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (input_width >= 4 && skip_scale >= 0 &&
      k1_fits(input_width, kernels, input0, output, skip)) {
    k1_skip_vector(input0, kernels, output, skip, input_width, input_channels,
                   output_channels, scale, skip_scale);
    return;
  }
#endif
  conv2dk1_skip_ui8_i8_i8_scalar(input0, kernels, output, skip, input_width,
                                 input_channels, output_channels, scale,
                                 skip_scale);
}

#endif // UNSIGNED_SKIP

#else // Vector

#endif // Vector
#endif // REGULAR

} // extern "C"
