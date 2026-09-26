//===- conv2dk1.cc -------------------------------------------------*- C++
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
#include <aie_api/aie.hpp>

#if AIE_TUNED_AIE2P
#include "bn_conv2dk1_aie2.h"
#endif

#define REL_WRITE 0
#define REL_READ 1

const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;

#if defined(BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH_NEW)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_ui8_scalar_input_split_partial_width_get_new(
    uint8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int32_t input_split, int32_t output_split,
    const int32_t weight_index, const int32_t x_start, const int32_t oc) {
  event0();
  int ic, ic8, oc8;

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

      // Transfer scalar sum to vector
      sum = current_sum;
      v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
    }
    // acc_cas=get_scd_v16acc64();
    // int scale_new=8;
    v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int sum_srs = 0;
      int cascade_sum = 0;
      sum = ext_elem(v16vec_partial[pixel], oc8);
      cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
      sum_srs = (((cascade_sum + sum) + (1 << (scale - 1)) - 1 +
                  (((cascade_sum + sum) >> scale) & 1)) >>
                 scale);
      sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

      oc_offset =
          oc + oc8_iter * (weight_index); // works fine when oc8_iter is 4
      output[(oc_offset * input_width * 8) + (pixel * 8) + oc8] = sum_srs;
    }
  }

  event1();
}
#endif

#if defined(BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_ui8_scalar_input_split_partial_width_get(
    uint8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int32_t input_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {
  event0();
  int ic, ic8, oc8;

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
        sum = ext_elem(v16vec_partial[pixel], oc8);
        cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
        sum_srs = (((sum + cascade_sum) + (1 << (scale - 1)) - 1 +
                    (((sum + cascade_sum) >> scale) & 1)) >>
                   scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * input_width * 8) + (pixel * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif

#if defined(PARTIAL_GET_I8_CAS_WIDTH_NEW) ||                                   \
    defined(BN13_2_PARTIAL_GET_I8_CAS_WIDTH_NEW) ||                            \
    defined(BN13_1_PARTIAL_GET_I8_CAS_WIDTH_NEW) ||                            \
    defined(BN14_1_PARTIAL_GET_I8_CAS_WIDTH_NEW)
static inline bool
k1_cas_get_new(int8_t *input, int8_t *kernels, uint8_t *output,
               const int32_t input_width, const int32_t input_channels,
               const int32_t output_channels, const int scale,
               const int32_t input_split, const int32_t output_split,
               const int32_t weight_index, const int32_t oc) {
#if AIE_TUNED_AIE2P
  if (!k1_cas_fits(kernels, input_split, output_split))
    return false;
  event0();
  const int32_t blocks = k1_per_split(input_channels, input_split) / 8;
  const int32_t row = input_width * 8;
  const int32_t oc_out =
      oc + k1_per_split(output_channels, output_split) / 8 * weight_index;
  k1_cas_get(input + blocks * row, kernels + oc * blocks * 64,
             output + oc_out * row, row, blocks,
             [=](auto &acc) { return acc.template to_vector<uint8>(scale); });
  event1();
  return true;
#else
  return false;
#endif
}

// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_i8_ui8_scalar_partial_width_get_new(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t output_split,
    int32_t weight_index, int32_t x_start, int32_t oc) {
  event0();
  int ic, ic8, oc8;

  int pixel_limit = 7;

  // Determine the start and end of the loop based on the chunk index for
  // weights
  int input_channel_chunk_size = input_channels / input_split;
  int start_ic = 1 * input_channel_chunk_size;
  int end_ic = start_ic + input_channel_chunk_size;
  int pixel = 0;
  int oc_offset = 0;
  int oc8_iter = output_channels / (8 * output_split);
  // Preload vector register with partial sums from previous iteration

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
                          ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }
      // Transfer scalar sum to vector
      sum = current_sum;
      v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
    }

    // if (end_ic == input_channels) {
    // acc_cas=get_scd_v16acc64();
    // int scale_new=8;
    v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int sum_srs = 0;
      int cascade_sum = 0;
      sum = ext_elem(v16vec_partial[pixel], oc8);
      cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
      // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
      sum_srs = (((sum + cascade_sum) + (1 << (scale - 1)) - 1 +
                  (((sum + cascade_sum) >> scale) & 1)) >>
                 scale);
      sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

      oc_offset =
          oc + oc8_iter * (weight_index); // works fine when oc8_iter is 4

      output[(oc_offset * input_width * 8) + (pixel * 8) + (oc8)] = sum_srs;
    }
    // }
  }

  event1();
}
#endif

#if defined(PARTIAL_GET_I8_CAS_WIDTH) ||                                       \
    defined(BN13_1_PARTIAL_GET_I8_CAS_WIDTH) ||                                \
    defined(BN14_1_PARTIAL_GET_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_i8_ui8_scalar_partial_width_get(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x_start,
    int32_t oc) {
  event0();
  int ic, ic8, oc8;

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

  if (weight_index >
      input_split / 2) { // Preload vector register with partial sum from
                         // previous iteration. If weight is only 1 then we
                         // don't have partial sum anyway
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
                          ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }
      // Extract the partial sum if applicable
      if (weight_index > input_split / 2) {
        last_sum = ext_elem(v16vec_partial[pixel], oc8);
      }
      // Transfer scalar sum to vector
      sum = current_sum + last_sum;
      v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
      if (input_split != (weight_index + 1)) {
        *accumulators[pixel] = lups(v16vec_partial[pixel], 0);
      }
    }

    if (end_ic == input_channels) {
      // acc_cas=get_scd_v16acc64();
      // int scale_new=8;
      v16vec_cas[pixel] = lsrs(get_scd_v16acc64(), 0, 0);
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;
        int cascade_sum = 0;
        sum = ext_elem(v16vec_partial[pixel], oc8);
        cascade_sum = ext_elem(v16vec_cas[pixel], oc8);
        // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
        sum_srs = (((sum + cascade_sum) + (1 << (scale - 1)) - 1 +
                    (((sum + cascade_sum) >> scale) & 1)) >>
                   scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * input_width * 8) + (pixel * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif

#ifdef PARTIAL_WIDTH
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to
// x_start + 8) simultaneously within each output channel (oc8 iteration).
//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_ui8_scalar_partial_width(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x_start,
    int32_t oc) {

  event0();
  int ic, ic8, oc8;

  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;

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

  for (oc8 = 0; oc8 < 8; oc8++) {
    int sum[8] = {0};
    int current_sum[8] = {0};
    int sum_srs[8] = {0};
    int last_sum[8] = {0};

    // Current iteration: go over all the input channels
    for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
      for (ic8 = 0; ic8 < 8; ic8++) {
        for (int pixel = 0; pixel < 8; pixel++) {
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
    if (weight_index != 0 && oc8 == 0) { // Preload vector register with partial
                                         // sum from previous iteration
      for (int pixel = 0; pixel < 8; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
        }
      }
    }

    if (weight_index != 0) { // Extract the partial sum
      for (int pixel = 0; pixel < 8; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          last_sum[pixel] = ext_elem(v16vec_partial[pixel], oc8);
        }
      }
    }

    for (int pixel = 0; pixel < 8; pixel++) {
      int x = x_start + pixel;
      if (x < input_width) {
        sum[pixel] = current_sum[pixel] + last_sum[pixel];

        // Transfer scalar sum to vector
        v16vec_partial[pixel] =
            upd_elem(v16vec_partial[pixel], oc8, sum[pixel]);
      }
    }

    if (end_ic == input_channels) { // if final set of input channels, scale the
                                    // final output
      for (int pixel = 0; pixel < 8; pixel++) {
        int x = x_start + pixel;
        if (x < input_width) {
          sum_srs[pixel] = (sum[pixel] + (1 << (scale - 1))) >> scale;
          sum_srs[pixel] = (sum_srs[pixel] > UMAX) ? UMAX
                           : (sum_srs[pixel] < 0)  ? 0
                                                   : sum_srs[pixel];
          output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs[pixel];
        }
      }
    }

    if (oc8 == 7) { // end of vectorization
      for (int pixel = 0; pixel < 8; pixel++) {
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

#ifdef PARTIAL_GET_I8_CAS
// Output Channel First Approach: Iterates over each output channel (oc) first
// and then processes all pixels (x) within that output channel iteration.
void conv2dk1_i8_ui8_scalar_partial_get(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x) {
  event0();
  int oc, ic, ic8, oc8;

  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;

  // Array of pointers to the accumulators
  v16acc64 *accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
      &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
      &v16acc_partial6, &v16acc_partial7, &v16acc_partial8};

  // static v16acc64 v16acc_partial;

  // Determine the start and end of the loop based on the chunk index for
  // weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;
  for (oc = 0; oc < output_channels / 8; oc++) {
    // for (x = 0; x < input_width; x++) { // col of output image
    v16acc64 &accumulator = *accumulators[oc % 9];
    v16int32 v16vec_partial = lsrs(accumulator, 0, 0);
    int value_index = 0;
    int cascade_sum = 0;
    v16acc64 acc_cas = undef_v16acc64(); // Get the accumulated values
    v16int32 vec_cas = undef_v16int32(); // Convert accumulator to vector

    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int sum_srs = 0;
      int last_sum = 0;

      if (oc8 == 0 &&
          end_ic == input_channels) { // if final set of input channels, scale
                                      // the final output
        // Get cascade sum
        acc_cas = get_scd_v16acc64();  // Get the accumulated values
        vec_cas = lsrs(acc_cas, 0, 0); // Convert accumulator to vector
      }

      // Current iteration: go over all the input channels
      for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          int val = input[(ic * input_width * 8) + (x * 8) + ic8];
          int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                          ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }

      if (weight_index != 1) { // Extract the partial sum
        last_sum = ext_elem(v16vec_partial, oc8);
      }

      sum = current_sum + last_sum;

      // Transfer scalar sum to vector
      v16vec_partial = upd_elem(v16vec_partial, oc8, sum);

      if (end_ic == input_channels) { // if final set of input channels, scale
                                      // the final output
        cascade_sum = ext_elem(vec_cas, oc8);
        sum_srs = ((sum + cascade_sum) + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }

      if (oc8 == 7) { // end of vectorization
        // // Transfer the values from vec to acc
        accumulator = lups(v16vec_partial, 0);
      }
    }
  }
  // }
  event1();
}

#endif

#ifdef PARTIAL
// Output Channel First Approach: Iterates over each output channel (oc) first
// and then processes all pixels (x) within that output channel iteration.
//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_ui8_scalar_partial(int8_t *input, int8_t *kernels,
                                    uint8_t *output, const int32_t input_width,
                                    const int32_t input_channels,
                                    const int32_t output_channels,
                                    const int scale, int32_t input_split,
                                    int32_t weight_index, int32_t x) {

  event0();
  int oc, ic, ic8, oc8;
  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;

  // Array of pointers to the accumulators
  v16acc64 *accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
      &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
      &v16acc_partial6, &v16acc_partial7, &v16acc_partial8};

  // static v16acc64 v16acc_partial;

  // Determine the start and end of the loop based on the chunk index for
  // weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;
  for (oc = 0; oc < output_channels / 8; oc++) {
    // for (x = 0; x < input_width; x++) { // col of output image
    v16acc64 &accumulator = *accumulators[oc % 9];
    v16int32 v16vec_partial = lsrs(accumulator, 0, 0);
    int value_index = 0;

    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int sum_srs = 0;
      int last_sum = 0;

      // Current iteration: go over all the input channels
      for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          int val = input[(ic * input_width * 8) + (x * 8) + ic8];
          int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                          ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
          current_sum += val * k;
        }
      }

      if (weight_index != 0) { // Extract the partial sum
        last_sum = ext_elem(v16vec_partial, value_index);
      }

      sum = current_sum + last_sum;

      // Transfer scalar sum to vector
      v16vec_partial = upd_elem(v16vec_partial, value_index, sum);
      value_index++;

      if (end_ic == input_channels) { // if final set of input channels, scale
                                      // the final output
        // Transfer the values from acc to vect
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }

      if (oc8 == 7) { // end of vectorization
        // // Transfer the values from vec to acc
        accumulator = lups(v16vec_partial, 0);
        value_index = 0;
      }
    }
  }
  // }
  event1();
}
#endif

//*****************************************************************************
// conv2d 1x1_GET - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

#ifdef GET
void conv2dk1_i8_ui8_scalar_cascade_get(
    int8_t *input0, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split, const int32_t weight_index, const int scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic =
      input_channels / 2 + weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
        if (weight_index == 0)
          sum[x] = 0;
        int sum_srs = 0;

        // Extract cascade sum values when starting a new block
        if (value_index == 0) {
          v16acc_partial = get_scd_v16acc64(); // Get the accumulated values
          v16vec_partial =
              lsrs(v16acc_partial, 0, 0); // Convert accumulator to vector
        }

        // Extract the specific cascade sum for the current index
        int partial_sum = ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = start_ic / 8; ic < end_ic / 8; ic++) {

          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                            ((ic - input_channel_chunk_size / 8) * 64) +
                            (ic8 * 8) + oc8];

            sum[x] += val * k;
          }
        }

        if (value_index == MAX_VALUES) {
          value_index = 0;
        }
        // scale for convolution
        sum[x] = sum[x] + partial_sum;
        // sum=partial_sum;
        if (end_ic == input_channels) {
          sum_srs = (sum[x] + (1 << (scaleT - 1))) >> scaleT;
          sum_srs = (sum_srs > UMAX) ? UMAX
                    : (sum_srs < 0)  ? 0
                                     : sum_srs; // clip
          // clip

          output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
        }
      }
    }
  }

  event1();
}
#endif

// #if defined (BN2)
// #ifdef INT8_ACT

// //*****************************************************************************
// // conv2d 1x1 - scalar
// // act: int8, wts: int8, out: uint8
// //*****************************************************************************
// void bn2_conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
//                         const int32_t input_width, const int32_t
//                         input_channels, const int32_t output_channels, const
//                         int scale) {
//   event0();

//   int x, ic, oc, ic8, oc8;
//   // scale=-17;
//   for (oc = 0; oc < output_channels / 8; oc++) {
//     for (x = 0; x < input_width; x++) { // col of output image
//       for (oc8 = 0; oc8 < 8; oc8++) {
//         int sum = 0;
//         int sum_srs = 0;

//         for (ic = 0; ic < input_channels / 8; ic++) {
//           for (ic8 = 0; ic8 < 8; ic8++) {
//             int val = input[(ic * input_width * 8) + (x * 8) + ic8];
//             int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
//                             (ic8 * 8) + oc8];
//             sum += val * k;
//           }
//         }

//         // sum_srs=sum>>scale;
//         sum_srs = (sum + (1 << (scale - 1))) >> scale;
//         sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
//         // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
//         output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//       }
//     }
//   }

//   event1();
// }
// #endif
// #endif

// #if defined (BN3)
// #ifdef INT8_ACT

// //*****************************************************************************
// // conv2d 1x1 - scalar
// // act: int8, wts: int8, out: uint8
// //*****************************************************************************
// void bn3_conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
//                         const int32_t input_width, const int32_t
//                         input_channels, const int32_t output_channels, const
//                         int scale) {
//   event0();

//   int x, ic, oc, ic8, oc8;
//   // scale=-17;
//   for (oc = 0; oc < output_channels / 8; oc++) {
//     for (x = 0; x < input_width; x++) { // col of output image
//       for (oc8 = 0; oc8 < 8; oc8++) {
//         int sum = 0;
//         int sum_srs = 0;

//         for (ic = 0; ic < input_channels / 8; ic++) {
//           for (ic8 = 0; ic8 < 8; ic8++) {
//             int val = input[(ic * input_width * 8) + (x * 8) + ic8];
//             int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
//                             (ic8 * 8) + oc8];
//             sum += val * k;
//           }
//         }

//         // sum_srs=sum>>scale;
//         sum_srs = (sum + (1 << (scale - 1))) >> scale;
//         sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
//         // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
//         output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//       }
//     }
//   }

//   event1();
// }
// #endif
// #endif

// #if defined (BN12)
// #ifdef INT8_ACT

// //*****************************************************************************
// // conv2d 1x1 - scalar
// // act: int8, wts: int8, out: uint8
// //*****************************************************************************
// void test_conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
//                         const int32_t input_width, const int32_t
//                         input_channels, const int32_t output_channels, const
//                         int scale) {
//   event0();

//   int x, ic, oc, ic8, oc8;
//   // scale=-17;
//   int applied_scale=scale;
//   for (oc = 0; oc < output_channels / 8; oc++) {
//     for (x = 0; x < input_width; x++) { // col of output image
//       for (oc8 = 0; oc8 < 8; oc8++) {
//         int32_t sum = 0;
//         int32_t sum_srs = 0;

//         for (ic = 0; ic < input_channels / 8; ic++) {
//           for (ic8 = 0; ic8 < 8; ic8++) {
//             int val = input[(ic * input_width * 8) + (x * 8) + ic8];
//             int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
//                             (ic8 * 8) + oc8];
//             sum += val * k;
//           }
//         }

//         // sum_srs=sum>>scale;
//         sum_srs = ((sum + (1 << (applied_scale - 1)) - 1 + ((sum >>
//         applied_scale) & 1)) >> applied_scale);
//         // sum_srs = (sum + (1 << (applied_scale - 1))) >> applied_scale;
//         sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
//         // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
//         output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//       }
//     }
//   }

//   event1();
// }
// #endif
// #endif

#if defined(POSTL2_PARTIAL)
#ifdef UINT16_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

static void
conv2dk1_ui16_partial_scalar(uint16_t *input, int8_t *kernels, int32_t *output,
                             const int32_t input_width,
                             const int32_t input_channels,
                             const int32_t output_channels, const int scale,
                             const int32_t ifm_index, const int32_t total_ifm) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  // int applied_scale=scale;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int32_t sum = 0;
        int32_t sum_srs = 0;
        int32_t accumulator =
            (ifm_index != 0) ? output[(oc * input_width * 8) + (x * 8) + oc8]
                             : 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        accumulator += sum;
        if (ifm_index == total_ifm - 1) {
          sum_srs = ((accumulator + (1 << (scale - 1)) - 1 +
                      ((accumulator >> scale) & 1)) >>
                     scale);
          sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

          output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
        } else {
          // Store the accumulated result in the output
          output[(oc * input_width * 8) + (x * 8) + oc8] = (int32_t)accumulator;
        }
      }
    }
  }
  event1();
}
#endif
#endif

#if defined(POSTL2_PARTIAL_ACC)
#ifdef UINT16_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

static void conv2dk1_ui16_partial_acc_scalar(
    uint16_t *input, int8_t *kernels, uint8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int scale, const int32_t ifm_index,
    const int32_t total_ifm) {
  event0();

  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;

  // Array of pointers to the accumulators
  v16acc64 *accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
      &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
      &v16acc_partial6, &v16acc_partial7, &v16acc_partial8};

  v16int32 v16vec_partial[6] = {}; // 80 elements

  int x, ic, oc, ic8, oc8;
  int32_t sum = 0;
  int32_t sum_srs = 0;
  // scale=-17;
  // int applied_scale=scale;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {

        sum = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // Calculate the index in the accumulators
        int accumulator_index = (oc * input_width * 8 + x * 8 + oc8) / 16;
        int element_index = (oc * input_width * 8 + x * 8 + oc8) % 16;
        int last_sum = 0;
        if (ifm_index > 0) {
          v16vec_partial[accumulator_index] =
              lsrs(*accumulators[accumulator_index], 0, 0);
          last_sum = ext_elem(v16vec_partial[accumulator_index], element_index);
        }
        // Store the result in the appropriate accumulator
        v16vec_partial[accumulator_index] = upd_elem(
            v16vec_partial[accumulator_index], element_index, last_sum + sum);
        if (element_index == 15)
          *accumulators[accumulator_index] =
              lups(v16vec_partial[accumulator_index], 0);
      }
      // output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
    }
  }

  // Transfer accumulator results to output if ifm_index is 5
  if (ifm_index == (total_ifm - 1)) {
    for (int i = 0; i < total_ifm; i++) {
      v16vec_partial[i] = lsrs(*accumulators[i], 0, 0);
      for (int j = 0; j < 16; j++) {

        int global_index = i * 16 + j;
        int oc = global_index / (input_width * 8);
        int remaining = global_index % (input_width * 8);
        int x = remaining / 8;
        int oc8 = remaining % 8;
        int output_index = (oc * input_width * 8) + (x * 8) + oc8;
        sum = ext_elem(v16vec_partial[i], j);
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        output[output_index] = sum_srs;
      }
    }
  }

  event1();
}

#endif
#endif
// was trying to do accumulation alive in vector register below. Updated the
// code above to keep results in buffer. static void
// conv2dk1_ui16_partial_scalar(uint16_t *input, int8_t *kernels, uint8_t
// *output,
//                         const int32_t input_width, const int32_t
//                         input_channels, const int32_t output_channels, const
//                         int scale,const int32_t ifm_index,const int32_t
//                         total_ifm) {
//   event0();

//     static v16acc64 v16acc_partial0;
//     static v16acc64 v16acc_partial1;
//     static v16acc64 v16acc_partial2;
//     static v16acc64 v16acc_partial3;
//     static v16acc64 v16acc_partial4;
//     static v16acc64 v16acc_partial5;
//     static v16acc64 v16acc_partial6;
//     static v16acc64 v16acc_partial7;
//     static v16acc64 v16acc_partial8;

//     // Array of pointers to the accumulators
//     v16acc64* accumulators[] = {
//         &v16acc_partial0, &v16acc_partial1, &v16acc_partial2,
//         &v16acc_partial3, &v16acc_partial4, &v16acc_partial5,
//         &v16acc_partial6, &v16acc_partial7, &v16acc_partial8
//     };

//   v16int32 v16vec_partial[6] = {}; // 80 elements

//   int x, ic, oc, ic8, oc8;
//   int32_t sum = 0;
//   int32_t sum_srs = 0;
//   // scale=-17;
//   // int applied_scale=scale;
//   for (oc = 0; oc < output_channels / 8; oc++) {
//     for (x = 0; x < input_width; x++) { // col of output image
//       for (oc8 = 0; oc8 < 8; oc8++) {

//         sum=0;
//         for (ic = 0; ic < input_channels / 8; ic++) {
//           for (ic8 = 0; ic8 < 8; ic8++) {
//             int val = input[(ic * input_width * 8) + (x * 8) + ic8];
//             int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
//                             (ic8 * 8) + oc8];
//             sum += val * k;
//           }
//         }

//         // Calculate the index in the accumulators
//         int accumulator_index = (oc * input_width * 8 + x * 8 + oc8) / 16;
//         int element_index = (oc * input_width * 8 + x * 8 + oc8) % 16;
//         int last_sum=0;
//         if(ifm_index>0)
//         {
//           v16vec_partial[accumulator_index] =
//           lsrs(*accumulators[accumulator_index],0,0); last_sum =
//           ext_elem(v16vec_partial[accumulator_index], element_index);
//         }
//         // Store the result in the appropriate accumulator
//         v16vec_partial[accumulator_index]=upd_elem(v16vec_partial[accumulator_index],
//         element_index, last_sum+sum); if(element_index==15)
//           *accumulators[accumulator_index] =
//           lups(v16vec_partial[accumulator_index],0);
//       }
//       // output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//     }
//   }

//   // Transfer accumulator results to output if ifm_index is 5
//     if (ifm_index == (total_ifm-1)) {
//         for (int i = 0; i < total_ifm; i++) {
//             v16vec_partial[i] = lsrs(*accumulators[i],0,0);
//             for (int j = 0; j < 16; j++) {

//                 int global_index = i * 16 + j;
//                 int oc = global_index / (input_width * 8);
//                 int remaining = global_index % (input_width * 8);
//                 int x = remaining / 8;
//                 int oc8 = remaining % 8;
//                 int output_index = (oc * input_width * 8) + (x * 8) + oc8;
//                 sum=ext_elem(v16vec_partial[i], j);
//                 sum_srs = ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) &
//                 1)) >> scale); sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs <
//                 0) ? 0 : sum_srs; output[output_index] = sum_srs;
//             }
//         }
//     }

//   event1();
// }

#if defined(POSTL2_PAD)
#ifdef UINT16_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
static void conv2dk1_ui16_scalar_pad(uint16_t *input, int8_t *kernels,
                                     uint16_t *output,
                                     const int32_t input_width,
                                     const int32_t input_channels,
                                     const int32_t input_channels_pad,
                                     const int32_t output_channels,
                                     const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  // int applied_scale=scale;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int32_t sum = 0;
        int32_t sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels_pad / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // sum_srs = (sum >> scale) << scale; // clip
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);

        // sum_srs = (sum + (1 << (applied_scale - 1))) >> applied_scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif
#endif

#if defined(POSTL2)
#ifdef UINT16_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
static void conv2dk1_ui16_scalar(uint16_t *input, int8_t *kernels,
                                 uint8_t *output, const int32_t input_width,
                                 const int32_t input_channels,
                                 const int32_t output_channels,
                                 const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  // int applied_scale=scale;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int32_t sum = 0;
        int32_t sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // sum_srs = (sum >> scale) << scale; // clip
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);

        // sum_srs = (sum + (1 << (applied_scale - 1))) >> applied_scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif
#endif

#if defined(BN13) || defined(BN14) || defined(REGULAR) || defined(POSTL1)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
static void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                               const int32_t input_width,
                               const int32_t input_channels,
                               const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  // int applied_scale=scale;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int32_t sum = 0;
        int32_t sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // sum_srs = (sum >> scale) << scale; // clip
        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);

        // sum_srs = (sum + (1 << (applied_scale - 1))) >> applied_scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif
#endif

#if defined(CONV_XPOOL_FUSED)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void fused_conv2dk1_x_pool_i8_scalar(int8_t *input, int8_t *kernels,
                                     uint16_t *output,
                                     const int32_t input_width,
                                     const int32_t input_channels,
                                     const int32_t output_channels,
                                     const int scale) {
  int x, ic, oc, ic8, oc8;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int32_t accumulator = 0;
      for (x = 0; x < input_width; x++) { // col of output image
        int32_t sum = 0;
        int32_t sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

        accumulator += sum_srs;
      }
      output[(oc * 1 * 8) + oc8] = accumulator;
    }
  }
}
#endif
#endif

#if defined(CONV_XYPOOL_FUSED)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void fused_conv2dk1_xy_pool_i8_scalar(int8_t *input, int8_t *kernels,
                                      uint16_t *output,
                                      const int32_t input_width,
                                      const int32_t input_channels,
                                      const int32_t output_channels,
                                      const int scale, const int y_index) {
  int x, ic, oc, ic8, oc8;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      // Initialize the accumulator
      uint16_t accumulator = (y_index != 0) ? output[(oc * 8) + oc8] : 0;

      for (x = 0; x < input_width; x++) { // Iterate over input columns (width)
        int32_t sum = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        int32_t sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

        accumulator += sum_srs;
      }

      if (y_index == input_width - 1) {
        float avg = accumulator / 49.0f;
        int rounded_avg = (int)(avg + 0.5f); // Round to the nearest integer
        output[(oc * 8) + oc8] = (uint16_t)rounded_avg;
      } else
        // Store the accumulated result in the output
        output[(oc * 8) + oc8] = (uint16_t)accumulator;
    }
  }
}
#endif
#endif

#if defined(CONV_XYPOOL_FUSED_LARGE_OUTPUT_CHANNEL_SPLIT)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void fused_conv2dk1_xy_pool_i8_large_output_channel_split_scalar(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int avgpool_scale, const int y_index,
    int32_t output_split, int32_t weight_index) {
  int x, ic, oc, ic8, oc8;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      // Initialize the accumulator

      uint32_t accumulator = (y_index != 0) ? output[(oc * 1 * 8) + oc8] : 0;

      for (x = 0; x < input_width; x++) { // Iterate over input columns (width)
        int32_t sum = 0;
        int32_t sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        accumulator += (uint8_t)sum_srs;
      }

      // Finalize the accumulator value if this is the last row
      if (y_index == input_width - 1) {
        float avg = accumulator / 49.0f;
        int rounded_avg;
        if ((int)(avg * 10) % 10 == 5) {
          rounded_avg = ((int)avg % 2 == 0) ? (int)avg : (int)avg + 1;
        } else {
          rounded_avg = (int)(avg + 0.5f);
        }
        // int32_t rounded_avg_srs = ((rounded_avg + (1 << (avgpool_scale - 1))
        // - 1 + ((rounded_avg >> avgpool_scale) & 1)) >> avgpool_scale);
        output[(oc * 1 * 8) + oc8] = (uint16_t)rounded_avg;
      } else {
        // Store the accumulated result in the output
        output[(oc * 1 * 8) + oc8] = (uint16_t)accumulator;
      }
    }
  }
}
#endif
#endif

#if defined(CONV_XYPOOL_FUSED_LARGE_PADDED)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void fused_conv2dk1_xy_pool_i8_large_padded_scalar(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t output_channels_padd, const int scale, const int y_index,
    int32_t output_split, int32_t weight_index) {
  int x, ic, oc, ic8, oc8;
  int oc_tile = output_channels / output_split;
  int oc_offset = oc_tile / 8 * weight_index;
  int padded_channels = (output_channels_padd > output_channels)
                            ? output_channels_padd - output_channels
                            : 0;

  for (oc = 0; oc < oc_tile / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      // Initialize the accumulator
      uint32_t accumulator =
          (y_index != 0) ? output[(oc_offset + oc) * 8 + oc8] : 0;

      for (x = 0; x < input_width; x++) { // Iterate over input columns (width)
        int32_t sum = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        int32_t sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

        accumulator += sum_srs;
      }

      // Finalize the accumulator value if this is the last row
      if (y_index == input_width - 1) {
        float avg = accumulator / 49.0f;
        int rounded_avg;
        if ((int)(avg * 10) % 10 == 5) {
          rounded_avg = ((int)avg % 2 == 0) ? (int)avg : (int)avg + 1;
        } else {
          rounded_avg = (int)(avg + 0.5f);
        }
        output[(oc_offset + oc) * 8 + oc8] = (uint16_t)rounded_avg;
      } else {
        // Store the accumulated result in the output
        output[(oc_offset + oc) * 8 + oc8] = (uint16_t)accumulator;
      }
    }
  }

  // Padding additional output channels with 0
  for (oc = output_channels; oc < output_channels_padd; oc++) {
    output[oc] = 0;
  }
}
#endif
#endif

#if defined(CONV_XYPOOL_FUSED_LARGE)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void fused_conv2dk1_xy_pool_i8_large_scalar(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int y_index, int32_t output_split,
    int32_t weight_index) {
  int x, ic, oc, ic8, oc8;
  int oc_tile = output_channels / output_split;
  int oc_offset = oc_tile / 8 * weight_index;

  for (oc = 0; oc < oc_tile / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      // Initialize the accumulator
      // uint16_t accumulator = (y_index != 0) ? output[(oc_offset + oc) * 8 +
      // oc8] : 0;
      uint32_t accumulator =
          (y_index != 0) ? output[(oc_offset + oc) * 8 + oc8] : 0;

      for (x = 0; x < input_width; x++) { // Iterate over input columns (width)
        int32_t sum = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        int32_t sum_srs =
            ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;

        accumulator += sum_srs;
      }

      // Finalize the accumulator value if this is the last row
      if (y_index == input_width - 1) {
        float avg = accumulator / 49.0f;
        int rounded_avg;
        if ((int)(avg * 10) % 10 == 5) {
          rounded_avg = ((int)avg % 2 == 0) ? (int)avg : (int)avg + 1;
        } else {
          rounded_avg = (int)(avg + 0.5f);
        }
        output[(oc_offset + oc) * 8 + oc8] = (uint16_t)rounded_avg;
      } else {
        // Store the accumulated result in the output
        output[(oc_offset + oc) * 8 + oc8] = (uint16_t)accumulator;
      }
    }
  }
}
#endif
#endif
// #if defined (BN1) ||(BN2) ||(BN3) ||(BN4) || (BN5) || (BN6) || (BN7) || (BN8)
// || (BN9) || (BN10) || (BN11) || (BN12) || (BN13) || (BN14) ||  (REGULAR)
// #ifdef UINT8_ACT
// //*****************************************************************************
// // conv2d 1x1 - scalar
// // act: uint8, wts: int8, out: uint8
// //*****************************************************************************
// void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, uint8_t *output,
//                          const int32_t input_width,
//                          const int32_t input_channels,
//                          const int32_t output_channels, const int scale) {
//   event0();

//   int x, ic, oc, ic8, oc8;
//   // scale=-17;
//   for (oc = 0; oc < output_channels / 8; oc++) {
//     for (x = 0; x < input_width; x++) { // col of output image
//       for (oc8 = 0; oc8 < 8; oc8++) {
//         int sum = 0;
//         int sum_srs = 0;

//         for (ic = 0; ic < input_channels / 8; ic++) {
//           for (ic8 = 0; ic8 < 8; ic8++) {
//             uint8_t val = input[(ic * input_width * 8) + (x * 8) + ic8];
//             int8_t k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
//                                (ic8 * 8) + oc8];
//             sum += val * k;
//           }
//         }

//         // sum_srs=sum>>scale;
//         // sum_srs = (sum + (1 << (scale - 1))) >> scale;
//         sum_srs = ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >>
//         scale); sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 :
//         sum_srs;
//         // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
//         output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//       }
//     }
//   }

//   event1();
// }

// #endif // UINT8_ACT
// #endif

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
#include "bn_conv2dk1_aie2.h"

// Rounds half to even and saturates like the scalar.
template <bool Aligned, int P = 4>
static void k1_relu_rows(const int8_t *input, const int8_t *kernels,
                         uint8_t *output, const int32_t input_width,
                         const int32_t input_channels,
                         const int32_t output_channels, const int scale) {
  k1_rows<Aligned, P>(
      input, kernels, output, input_width, input_channels, output_channels,
      [=](auto &acc) { return acc.template to_vector<uint8>(scale); });
}

template <int P>
static void k1_relu_chunked(const int8_t *input, const int8_t *kernels,
                            uint8_t *output, const int32_t input_width,
                            const int32_t input_channels,
                            const int32_t output_channels, const int scale) {
  if (input_width % P == 0 &&
      (((uintptr_t)input | (uintptr_t)output) & (8 * P - 1)) == 0)
    k1_relu_rows<true, P>(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
  else
    k1_relu_rows<false, P>(input, kernels, output, input_width, input_channels,
                           output_channels, scale);
}

static void k1_vector(const int8_t *input, const int8_t *kernels,
                      uint8_t *output, const int32_t input_width,
                      const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
#if defined(K1_WIDTH)
  k1_relu_rows<K1_ALIGNED, K1_P>(input, kernels, output, K1_WIDTH,
                                 input_channels, output_channels, scale);
#elif AIE_TUNED_AIE2P
  k1_relu_rows<true>(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
#else
  k1_relu_chunked<4>(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
#endif
  event1();
}

constexpr int32_t K1_POOL_MAX_WIDTH = 32;

// Out of line on AIE2P, where the pooled sums' divide constants otherwise stay
// live across the conv and spill, and the kernel's stack outgrows the default.
#if AIE_TUNED_AIE2P
#define K1_POOL_NOINLINE __attribute__((noinline))
#else
#define K1_POOL_NOINLINE
#endif

K1_POOL_NOINLINE static void k1_pool_pad(uint16_t *output, const int32_t start,
                                         const int32_t end) {
#if AIE_TUNED_AIE2P
  if (((uintptr_t)(output + start) & 31) == 0 && ((end - start) & 15) == 0) {
    aie::vector<uint16, 16> *p = (aie::vector<uint16, 16> *)(output + start);
    for (int32_t i = (end - start) / 16; i > 0; i--)
      *p++ = aie::zeros<uint16, 16>();
    return;
  }
#endif
  for (int32_t c = start; c < end; c++)
    output[c] = 0;
}

K1_POOL_NOINLINE static aie::vector<int32, 16>
k1_pool_avg(const aie::vector<int32, 16> res) {
  // (acc * 42799) >> 21 as (acc << 16) - acc * 22737, rounded down.
  aie::accum<acc64, 16> m;
  m.from_vector(res, 16);
  m = aie::mac(m, res, aie::broadcast<int16, 16>(-22737));
  aie::set_rounding(aie::rounding_mode::floor);
  const aie::vector<int32, 16> q = m.template to_vector<int32>(21);
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<int32, 16> r = aie::sub(
      res,
      aie::mul(q, aie::broadcast<int16, 16>(49)).template to_vector<int32>(0));
  const auto up = aie::ge(r, 30) |
                  (aie::ge(r, 25) &
                   aie::eq(aie::bit_and(q, aie::broadcast<int32, 16>(1)), 1));
  return aie::select(q, aie::add(q, 1), up);
}

// Pools one output channel block at a time. The requantized conv row goes to
// a stack buffer, where the overlapping last chunk just rewrites the same
// pixels and the bytes past the row stay zero, and is summed 4 pixels at a
// time. The accumulate follows the scalar, and the float average is replaced
// by its integer form: q + 1 when the remainder is at least 30, or 25-29 with
// q odd. The divide by 49 is a 32-bit multiply-shift. For sums below 100353
// (the largest here is 65535 + 255 * K1_POOL_MAX_WIDTH) it is one low only on
// multiples of 49, where the remainder 49 still rounds q up to the quotient.
K1_POOL_NOINLINE static void
k1_xy_pool_vector(const int8_t *input, const int8_t *kernels, uint16_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels,
                  const int32_t output_channels_padd, const int scale,
                  const int y_index, int32_t output_split,
                  int32_t weight_index) {
  alignas(32) uint8_t row[K1_POOL_MAX_WIDTH * 8];
  for (int i = 0; i < K1_POOL_MAX_WIDTH * 8; i += 32)
    aie::store_v(row + i, aie::zeros<uint8, 32>());
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int32_t oc_tile = output_channels / output_split;
  const int32_t oc_offset = oc_tile / 8 * weight_index;
  const bool aligned = input_width % 4 == 0 && ((uintptr_t)input & 31) == 0;
  const aie::vector<uint16, 8> prior =
      aie::broadcast<uint16, 8>(y_index != 0 ? 0xffffu : 0u);
  const bool last = y_index == input_width - 1;
  for (int oc = 0; oc < oc_tile / 8; oc++) {
    const int8_t *wts = kernels + oc * (input_channels / 8) * 64;
    if (aligned)
      k1_relu_rows<true>(input, wts, row, input_width, input_channels, 8,
                         scale);
    else
      k1_relu_rows<false>(input, wts, row, input_width, input_channels, 8,
                          scale);
    aie::vector<uint16, 32> sum = aie::zeros<uint16, 32>();
    for (int x = 0; x < input_width * 8; x += 32)
      sum = aie::add(sum, aie::load_v<32>(row + x).unpack());
    const aie::vector<uint16, 16> s16 =
        aie::add(sum.extract<16>(0), sum.extract<16>(1));
    uint16_t *o = output + (oc_offset + oc) * 8;
#if AIE_TUNED_AIE2P
    // Peano can't legalize a 128-bit vector and on AIE2P, so mask unpacked.
    const auto above = aie::bit_and(aie::load_v<8>(o).unpack(), prior.unpack());
#else
    const auto above = aie::bit_and(aie::load_v<8>(o), prior).unpack();
#endif
    const aie::vector<int32, 8> acc =
        aie::add(aie::vector_cast<int32>(above),
                 aie::vector_cast<int32>(
                     aie::add(s16.extract<8>(0), s16.extract<8>(1)).unpack()));
    aie::vector<int32, 16> res = aie::concat(acc, aie::zeros<int32, 8>());
    if (last)
      res = k1_pool_avg(res);
    aie::store_v(o, aie::filter_even(aie::vector_cast<uint16>(res), 1)
                        .template extract<8>(0));
  }
  k1_pool_pad(output, output_channels, output_channels_padd);
}

#if AIE_TUNED_AIE2P
// A row of at most 8 pixels is one mmul<8,8,8> per input channel block, the
// pixels past the row masked off once requantized, so no row goes through
// memory. G output channel blocks share each input load. The last input
// block is loaded ending at the input's end and shifted down, so no load
// reads past it. Needs at least 3 input channel blocks.
template <int G>
static inline void
k1_pool_group(const int8_t *__restrict in, const int8_t *__restrict wts,
              const int8_t *end, uint16_t *o, const int32_t row,
              const int32_t ic_blocks, const int scale,
              const aie::mask<64> keep, const aie::vector<uint16, 8> prior,
              const bool last) {
  const int32_t blk = ic_blocks * 64;
  aie::mmul<8, 8, 8, int8, int8> acc[G];
  aie::vector<int8, 64> a = k1_load<false, 64>(in);
  AIE_LOOP_UNROLL_FULL
  for (int g = 0; g < G; g++)
    acc[g].mul(a, aie::load_v<64>(wts + g * blk));
#pragma clang loop min_iteration_count(1)
  for (int ic = 2; ic < ic_blocks; ic++) {
    in += row;
    wts += 64;
    a = k1_load<false, 64>(in);
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < G; g++)
      acc[g].mac(a, aie::load_v<64>(wts + g * blk));
  }
  a = aie::shuffle_down(k1_load<false, 64>(end), 64 - row);
  aie::vector<int32, 8> sums[G];
  AIE_LOOP_UNROLL_FULL
  for (int g = 0; g < G; g++) {
    acc[g].mac(a, aie::load_v<64>(wts + 64 + g * blk));
    const aie::vector<uint8, 64> v = aie::select(
        aie::zeros<uint8, 64>(), acc[g].template to_vector<uint8>(scale), keep);
    const aie::vector<uint16, 32> s =
        aie::add(v.extract<32>(0).unpack(), v.extract<32>(1).unpack());
    const aie::vector<uint16, 16> s16 =
        aie::add(s.extract<16>(0), s.extract<16>(1));
    const auto above =
        aie::bit_and(aie::load_v<8>(o + g * 8).unpack(), prior.unpack());
    sums[g] =
        aie::add(aie::vector_cast<int32>(above),
                 aie::vector_cast<int32>(
                     aie::add(s16.extract<8>(0), s16.extract<8>(1)).unpack()));
  }
  AIE_LOOP_UNROLL_FULL
  for (int g = 0; g < G; g += 2) {
    aie::vector<int32, 16> res =
        aie::concat(sums[g], g + 1 < G ? sums[g + 1] : aie::zeros<int32, 8>());
    if (last)
      res = k1_pool_avg(res);
    const aie::vector<uint16, 16> r =
        aie::filter_even(aie::vector_cast<uint16>(res), 1);
    aie::store_v(o + g * 8, r.extract<8>(0));
    if (g + 1 < G)
      aie::store_v(o + g * 8 + 8, r.extract<8>(1));
  }
}

K1_POOL_NOINLINE static void
k1_xy_pool_narrow(const int8_t *input, const int8_t *kernels, uint16_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels,
                  const int32_t output_channels_padd, const int scale,
                  const int y_index, int32_t output_split,
                  int32_t weight_index) {
  constexpr int G = 5;
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int32_t blocks = output_channels / output_split / 8;
  const int32_t row = input_width * 8;
  const int32_t ic_blocks = input_channels / 8;
  const aie::vector<uint16, 8> prior =
      aie::broadcast<uint16, 8>(y_index != 0 ? 0xffffu : 0u);
  const bool last = y_index == input_width - 1;
  const aie::mask<64> keep =
      aie::mask<64>::from_uint64(input_width == 8 ? ~0ull : (1ull << row) - 1);
  const int8_t *end = input + ic_blocks * row - 64;
  uint16_t *o = output + blocks * weight_index * 8;
  int oc = 0;
  for (; oc + G <= blocks; oc += G)
    k1_pool_group<G>(input, kernels + oc * ic_blocks * 64, end, o + oc * 8, row,
                     ic_blocks, scale, keep, prior, last);
  for (; oc < blocks; oc++)
    k1_pool_group<1>(input, kernels + oc * ic_blocks * 64, end, o + oc * 8, row,
                     ic_blocks, scale, keep, prior, last);
  k1_pool_pad(output, output_channels, output_channels_padd);
}
#endif

#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

#if AIE_TUNED_AIE2
constexpr uintptr_t FC_ALIGN = 32;

// Fully connected: a single pixel, so the input is input_channels contiguous
// uint16 and each output channel block's weights are [input_channels][8].
// mmul<2,8,8> takes 16 inputs as a 2 x 8 matrix against the next 8 rows of
// weights; row 0 of acc0 and row 1 of acc1 are the ones that line up, and
// their sum is the dot product.
static void fc_ui16_vector(const uint16_t *__restrict input,
                           const int8_t *__restrict kernels,
                           uint16_t *__restrict output,
                           const int32_t input_channels,
                           const int32_t input_channels_pad,
                           const int32_t output_channels, const int scale) {
  using MMUL = aie::mmul<2, 8, 8, uint16, int8>;
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  const uint32_t pairs = (uint32_t)input_channels / 32;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const aie::vector<uint16, 16> *__restrict x =
        (const aie::vector<uint16, 16> *)input;
    const aie::vector<int8, 64> *__restrict w =
        (const aie::vector<int8, 64> *)(kernels +
                                        oc * (input_channels_pad / 8) * 64);
    MMUL acc0, acc1, acc2, acc3;
    acc0.mul(x[0], w[0]);
    acc1.mul(x[0], w[1]);
    acc2.mul(x[1], w[2]);
    acc3.mul(x[1], w[3]);
#pragma clang loop min_iteration_count(1)
    for (uint32_t c = 1; c < pairs; c++) {
      x += 2;
      w += 4;
      acc0.mac(x[0], w[0]);
      acc1.mac(x[0], w[1]);
      acc2.mac(x[1], w[2]);
      acc3.mac(x[1], w[3]);
    }
    x += 2;
    w += 4;
    aie::vector<int32, 16> s02 = aie::add(acc0.template to_vector<int32>(0),
                                          acc2.template to_vector<int32>(0));
    aie::vector<int32, 16> s13 = aie::add(acc1.template to_vector<int32>(0),
                                          acc3.template to_vector<int32>(0));
    if ((uint32_t)input_channels & 16) {
      MMUL t0, t1;
      t0.mul(x[0], w[0]);
      t1.mul(x[0], w[1]);
      s02 = aie::add(s02, t0.template to_vector<int32>(0));
      s13 = aie::add(s13, t1.template to_vector<int32>(0));
    }
    const aie::vector<int32, 8> sum =
        aie::add(s02.extract<8>(0), s13.extract<8>(1));
    aie::accum<acc32, 16> a;
    a.from_vector(aie::concat(sum, aie::zeros<int32, 8>()), 0);
    aie::store_v(output + oc * 8,
                 a.template to_vector<uint8>(scale).unpack().extract<8>(0));
  }
  event1();
}
#elif AIE_TUNED_AIE2P
constexpr uintptr_t FC_ALIGN = 64;

// As above, but AIE2P's smallest dense uint16 x int8 mmul is 4 x 8 x 8: 32
// inputs as a 4 x 8 matrix, with row j of acc j lining up with weight rows
// 8j to 8j + 7.
static void fc_ui16_vector(const uint16_t *__restrict input,
                           const int8_t *__restrict kernels,
                           uint16_t *__restrict output,
                           const int32_t input_channels,
                           const int32_t input_channels_pad,
                           const int32_t output_channels, const int scale) {
  using MMUL = aie::mmul<4, 8, 8, uint16, int8>;
  event0();
  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::conv_even);
  const uint32_t quads = (uint32_t)input_channels / 32;
  for (int oc = 0; oc < output_channels / 8; oc++) {
    const aie::vector<uint16, 32> *__restrict x =
        (const aie::vector<uint16, 32> *)input;
    const aie::vector<int8, 64> *__restrict w =
        (const aie::vector<int8, 64> *)(kernels +
                                        oc * (input_channels_pad / 8) * 64);
    MMUL acc0, acc1, acc2, acc3;
    acc0.mul(x[0], w[0]);
    acc1.mul(x[0], w[1]);
    acc2.mul(x[0], w[2]);
    acc3.mul(x[0], w[3]);
#pragma clang loop min_iteration_count(1)
    for (uint32_t c = 1; c < quads; c++) {
      x += 1;
      w += 4;
      acc0.mac(x[0], w[0]);
      acc1.mac(x[0], w[1]);
      acc2.mac(x[0], w[2]);
      acc3.mac(x[0], w[3]);
    }
    x += 1;
    w += 4;
    if ((uint32_t)input_channels & 16) {
      const aie::vector<uint16, 32> t = aie::concat(
          aie::load_v<16>((const uint16_t *)x), aie::zeros<uint16, 16>());
      acc0.mac(t, w[0]);
      acc1.mac(t, w[1]);
    }
    const aie::vector<int32, 32> v0 = acc0.template to_vector<int32>(0);
    const aie::vector<int32, 32> v1 = acc1.template to_vector<int32>(0);
    const aie::vector<int32, 32> v2 = acc2.template to_vector<int32>(0);
    const aie::vector<int32, 32> v3 = acc3.template to_vector<int32>(0);
    const aie::vector<int32, 8> sum =
        aie::add(aie::add(v0.extract<8>(0), v1.extract<8>(1)),
                 aie::add(v2.extract<8>(2), v3.extract<8>(3)));
    aie::accum<acc32, 16> a;
    a.from_vector(aie::concat(sum, aie::zeros<int32, 8>()), 0);
    aie::store_v(output + oc * 8,
                 a.template to_vector<uint8>(scale).unpack().extract<8>(0));
  }
  event1();
}
#endif // AIE_TUNED_AIE2 || AIE_TUNED_AIE2P

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH_NEW

void bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_get_new(
    uint8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int32_t input_split, int32_t output_split,
    const int32_t weight_index, const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_ui8_scalar_input_split_partial_width_get_new(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, output_split, weight_index, x_start, oc);
}
#endif

#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH

void bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_get(
    uint8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int32_t input_split, const int32_t weight_index,
    const int32_t x_start, const int32_t oc) {

  conv2dk1_ui8_ui8_scalar_input_split_partial_width_get(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, weight_index, x_start, oc);
}
#endif

#ifdef BN13_1_PARTIAL_GET_I8_CAS_WIDTH

void bn13_1_conv2dk1_i8_ui8_partial_width_get(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x_start,
    int32_t oc) {

  conv2dk1_i8_ui8_scalar_partial_width_get(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, weight_index, x_start, oc);
}
#endif

#ifdef BN14_1_PARTIAL_GET_I8_CAS_WIDTH

void bn14_1_conv2dk1_i8_ui8_partial_width_get(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x_start,
    int32_t oc) {

  conv2dk1_i8_ui8_scalar_partial_width_get(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, weight_index, x_start, oc);
}
#endif

#ifdef BN14_1_PARTIAL_GET_I8_CAS_WIDTH_NEW
void bn14_1_conv2dk1_i8_ui8_partial_width_get_new(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t output_split,
    int32_t weight_index, int32_t x_start, int32_t oc) {

  if (k1_cas_get_new(input, kernels, output, input_width, input_channels,
                     output_channels, scale, input_split, output_split,
                     weight_index, oc))
    return;
  conv2dk1_i8_ui8_scalar_partial_width_get_new(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, output_split, weight_index, x_start, oc);
}
#endif

#ifdef BN13_1_PARTIAL_GET_I8_CAS_WIDTH_NEW
void bn13_1_conv2dk1_i8_ui8_partial_width_get_new(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t output_split,
    int32_t weight_index, int32_t x_start, int32_t oc) {

  if (k1_cas_get_new(input, kernels, output, input_width, input_channels,
                     output_channels, scale, input_split, output_split,
                     weight_index, oc))
    return;
  conv2dk1_i8_ui8_scalar_partial_width_get_new(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, output_split, weight_index, x_start, oc);
}
#endif

#ifdef BN13_2_PARTIAL_GET_I8_CAS_WIDTH_NEW
void bn13_2_conv2dk1_i8_ui8_partial_width_get_new(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t output_split,
    int32_t weight_index, int32_t x_start, int32_t oc) {

  if (k1_cas_get_new(input, kernels, output, input_width, input_channels,
                     output_channels, scale, input_split, output_split,
                     weight_index, oc))
    return;
  conv2dk1_i8_ui8_scalar_partial_width_get_new(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, output_split, weight_index, x_start, oc);
}
#endif

#ifdef PARTIAL_GET_I8_CAS_WIDTH_NEW

void conv2dk1_i8_ui8_partial_width_get_new(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t output_split,
    int32_t weight_index, int32_t x_start, int32_t oc) {

  if (k1_cas_get_new(input, kernels, output, input_width, input_channels,
                     output_channels, scale, input_split, output_split,
                     weight_index, oc))
    return;
  conv2dk1_i8_ui8_scalar_partial_width_get_new(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, output_split, weight_index, x_start, oc);
}
#endif

#ifdef PARTIAL_GET_I8_CAS_WIDTH

void conv2dk1_i8_ui8_partial_width_get(
    int8_t *input, int8_t *kernels, uint8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, int32_t input_split, int32_t weight_index, int32_t x_start,
    int32_t oc) {

  conv2dk1_i8_ui8_scalar_partial_width_get(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, input_split, weight_index, x_start, oc);
}
#endif

// #ifdef BN10

//     #ifdef INT8_ACT

//     void bn10_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
//                     const int32_t input_width, const int32_t input_channels,
//                     const int32_t output_channels, const int scale) {
//       conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
//                         output_channels, scale);
//     }

//     #else // UINT8_ACT

//     void bn10_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
//                       const int32_t input_width, const int32_t
//                       input_channels, const int32_t output_channels, const
//                       int scale) {
//       conv2dk1_ui8_scalar(input, kernels, output, input_width,
//       input_channels,
//                           output_channels, scale);
//     }

//     #endif // UINT8_ACT

//     #endif // Vector
// #ifdef BN12

//     #ifdef INT8_ACT

//     void bn12_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
//                     const int32_t input_width, const int32_t input_channels,
//                     const int32_t output_channels, const int scale) {
//       conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
//                         output_channels, scale);
//     }

//     #else // UINT8_ACT

//     void bn12_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
//                       const int32_t input_width, const int32_t
//                       input_channels, const int32_t output_channels, const
//                       int scale) {
//       conv2dk1_ui8_scalar(input, kernels, output, input_width,
//       input_channels,
//                           output_channels, scale);
//     }

//     #endif // UINT8_ACT

// #endif // Vector

// #ifdef BN11

//       #ifdef INT8_ACT

//       void bn11_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
//                       const int32_t input_width, const int32_t
//                       input_channels, const int32_t output_channels, const
//                       int scale) {
//         conv2dk1_i8_scalar(input, kernels, output, input_width,
//         input_channels,
//                           output_channels, scale);
//       }

//       #else // UINT8_ACT

//       void bn11_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t
//       *output,
//                         const int32_t input_width, const int32_t
//                         input_channels, const int32_t output_channels, const
//                         int scale) {
//         conv2dk1_ui8_scalar(input, kernels, output, input_width,
//         input_channels,
//                             output_channels, scale);
//       }

//       #endif // UINT8_ACT

// #endif

#ifdef POSTL1
void post_L1_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels,
                                  uint8_t *output, const int32_t input_width,
                                  const int32_t input_channels,
                                  const int32_t output_channels,
                                  const int scale) {

  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

#endif

#ifdef POSTL2_PARTIAL_ACC
void post_L2_conv2dk1_relu_i16_ui8(uint16_t *input, int8_t *kernels,
                                   uint8_t *output, const int32_t input_width,
                                   const int32_t input_channels,
                                   const int32_t output_channels,
                                   const int scale, const int ifm_index,
                                   const int32_t total_ifm) {

  conv2dk1_ui16_partial_acc_scalar(input, kernels, output, input_width,
                                   input_channels, output_channels, scale,
                                   ifm_index, total_ifm);
}

#endif

#ifdef POSTL2_PARTIAL
void post_L2_conv2dk1_relu_i16_ui8(uint16_t *input, int8_t *kernels,
                                   int32_t *output, const int32_t input_width,
                                   const int32_t input_channels,
                                   const int32_t output_channels,
                                   const int scale, const int ifm_index,
                                   const int32_t total_ifm) {

  conv2dk1_ui16_partial_scalar(input, kernels, output, input_width,
                               input_channels, output_channels, scale,
                               ifm_index, total_ifm);
}

#endif

#ifdef POSTL2_PAD
void post_L2_conv2dk1_relu_i16_ui16_pad(uint16_t *input, int8_t *kernels,
                                        uint16_t *output,
                                        const int32_t input_width,
                                        const int32_t input_channels,
                                        const int32_t input_channels_pad,
                                        const int32_t output_channels,
                                        const int scale) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (input_width == 1 && input_channels >= 64 && input_channels % 16 == 0 &&
      (((uintptr_t)input | (uintptr_t)kernels) & (FC_ALIGN - 1)) == 0 &&
      ((uintptr_t)output & 15) == 0) {
    fc_ui16_vector(input, kernels, output, input_channels, input_channels_pad,
                   output_channels, scale);
    return;
  }
#endif
  conv2dk1_ui16_scalar_pad(input, kernels, output, input_width, input_channels,
                           input_channels_pad, output_channels, scale);
}

#endif

#ifdef POSTL2
void post_L2_conv2dk1_relu_i16_ui8(uint16_t *input, int8_t *kernels,
                                   uint8_t *output, const int32_t input_width,
                                   const int32_t input_channels,
                                   const int32_t output_channels,
                                   const int scale) {

  conv2dk1_ui16_scalar(input, kernels, output, input_width, input_channels,
                       output_channels, scale);
}

#endif

#ifdef CONV_XPOOL_FUSED
void conv2dk1_x_pool_fused_relu_i8_ui8(int8_t *input, int8_t *kernels,
                                       uint16_t *output,
                                       const int32_t input_width,
                                       const int32_t input_channels,
                                       const int32_t output_channels,
                                       const int scale) {

  fused_conv2dk1_x_pool_i8_scalar(input, kernels, output, input_width,
                                  input_channels, output_channels, scale);
}

#endif

#ifdef CONV_XYPOOL_FUSED
void conv2dk1_xy_pool_fused_relu_i8_ui8(int8_t *input, int8_t *kernels,
                                        uint16_t *output,
                                        const int32_t input_width,
                                        const int32_t input_channels,
                                        const int32_t output_channels,
                                        const int scale, const int y_index) {

  fused_conv2dk1_xy_pool_i8_scalar(input, kernels, output, input_width,
                                   input_channels, output_channels, scale,
                                   y_index);
}

#endif

#ifdef CONV_XYPOOL_FUSED_LARGE_PADDED
void conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t output_channels_padd, const int scale, const int y_index,
    int32_t output_split, int32_t weight_index) {
  event0();
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (input_width >= 4 && input_width <= K1_POOL_MAX_WIDTH &&
      ((uintptr_t)output & 15) == 0 && k1_wts_aligned(kernels)) {
#if AIE_TUNED_AIE2P
    if (input_width <= 8 && input_channels >= 24)
      k1_xy_pool_narrow(input, kernels, output, input_width, input_channels,
                        output_channels, output_channels_padd, scale, y_index,
                        output_split, weight_index);
    else
#endif
      k1_xy_pool_vector(input, kernels, output, input_width, input_channels,
                        output_channels, output_channels_padd, scale, y_index,
                        output_split, weight_index);
    event1();
    return;
  }
#endif
  fused_conv2dk1_xy_pool_i8_large_padded_scalar(
      input, kernels, output, input_width, input_channels, output_channels,
      output_channels_padd, scale, y_index, output_split, weight_index);
  event1();
}

#endif

#ifdef CONV_XYPOOL_FUSED_LARGE
void conv2dk1_xy_pool_fused_relu_large_i8_ui8(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int y_index, int32_t output_split,
    int32_t weight_index) {

  fused_conv2dk1_xy_pool_i8_large_scalar(input, kernels, output, input_width,
                                         input_channels, output_channels, scale,
                                         y_index, output_split, weight_index);
}

#endif

#ifdef CONV_XYPOOL_FUSED_LARGE_OUTPUT_CHANNEL_SPLIT
void conv2dk1_xy_pool_fused_relu_large_output_channel_split_i8_ui8(
    int8_t *input, int8_t *kernels, uint16_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int scale, const int avgpool_scale, const int y_index,
    int32_t output_split, int32_t weight_index) {

  fused_conv2dk1_xy_pool_i8_large_output_channel_split_scalar(
      input, kernels, output, input_width, input_channels, output_channels,
      scale, avgpool_scale, y_index, output_split, weight_index);
}

#endif
#ifdef REGULAR
void conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                          const int32_t input_width,
                          const int32_t input_channels,
                          const int32_t output_channels, const int scale) {
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  if (input_width >= 4 && k1_fits(input_width, kernels, input, output)) {
    k1_vector(input, kernels, output, input_width, input_channels,
              output_channels, scale);
    return;
  }
#endif
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}
#endif

} // extern "C"
