//===- conv2dk1_skip_init.cc -------------------------------------------------*-
// C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
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

#include <aie_api/aie.hpp>



const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;

// #define INT8_MAX 127
// #define INT8_MIN -128



#if defined (BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH) || (BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip, 
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale, const int skip_scale,
                                              const int32_t input_split, const int32_t weight_index, const int32_t x_start, const int32_t oc) {
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
    int pixel_limit=7;

    // Array of pointers to the accumulators
    v16acc64* accumulators[] = {
        &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
        &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
        &v16acc_partial8
    };


    // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

    // Determine the start and end of the loop based on the chunk index for weights
    int input_channel_chunk_size = input_channels / input_split;
    int start_ic = weight_index * input_channel_chunk_size;
    int end_ic =  start_ic + input_channel_chunk_size;
    int pixel = 0;
    // Preload vector register with partial sums from previous iteration
    // v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
    //                            undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    // v16int32 v16vec_cas[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
    //                            undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    v16acc64 acc_cas = undef_v16acc64();
    v16int32 v16vec_partial[8] = {}; 
    v16int32 v16vec_cas[8] = {}; 

    if (weight_index != 0 ) {  // Preload vector register with partial sum from previous iteration. If weight is only 1 then we don't have partial sum anyway
        for ( pixel = 0; pixel < pixel_limit; pixel++) {
            int x = x_start + pixel;
            if (x < input_width) {
                    v16vec_partial[pixel] = lsrs(*accumulators[pixel],0,0); 
            } 
        }
    }

    // Process each pixel across all output channels
    for (pixel = 0; pixel < pixel_limit; pixel++) {
        // Loop over output channels (oc8)
            for (oc8 = 0; oc8 < 8; oc8++) {
                int sum=0;
                int current_sum = 0;
                int last_sum= 0;
                int final_sum = 0;
                
                    
                // Loop over input channels in chunks of 8
                for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
                    for (ic8 = 0; ic8 < 8; ic8++) {
                        // int k_base = (0 * (input_channel_chunk_size / 8) * 64) + ((ic - start_ic / 8) * 64) + (ic8 * 8);
                            int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
                            int k = kernels[(0 * (input_channel_chunk_size / 8) * 64) +
                                            ((ic) * 64) + (ic8 * 8) + oc8];
                            current_sum+= val * k;
                    }
                }
                // Extract the partial sum if applicable
                if (weight_index != 0) {
                        last_sum = ext_elem(v16vec_partial[pixel], oc8);
                }
                // Transfer scalar sum to vector
                sum= current_sum + last_sum;
                v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
                if (weight_index != (input_split / 2 - 1)) {
                    *accumulators[pixel] = lups(v16vec_partial[pixel],0);
                }
            }

            if (weight_index == (input_split / 2 - 1)) {
                // acc_cas=get_scd_v16acc64();
                // int scale_new=8;
                v16vec_cas[pixel] = lsrs(get_scd_v16acc64(),0, 0);
                for (oc8 = 0; oc8 < 8; oc8++) {
                    int sum = 0;
                    int sum_srs = 0;
                    int cascade_sum = 0;
                    int skip_temp = 0;
                    int32_t skip_sum = 0;
                    int skip_sum_srs_final = 0;
                    int skip_sum_srs_final_out = 0;
                    sum = ext_elem(v16vec_partial[pixel], oc8);
                    cascade_sum=ext_elem(v16vec_cas[pixel], oc8);
                    sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs= (sum_srs > MAX) ? MAX : (sum_srs< -MAX) ? -MIN : sum_srs;
                    
                    skip_temp = skip[(oc * input_width * 8) + (pixel * 8) + oc8];
                    skip_sum = sum_srs + skip_temp;

                    skip_sum_srs_final = (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT; 
                    skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX : (skip_sum_srs_final < -MAX) ? -MIN : skip_sum_srs_final; // clip
                    
                    output[(oc * input_width * 8) + (pixel * 8) + oc8] = skip_sum_srs_final_out;
            }
        }       
    } 

    event1();
}
#endif


#if defined  (BN13_2_PARTIAL_GET_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(uint8_t *input, int8_t *kernels, uint8_t *output,int8_t *skip, 
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
    int pixel_limit=7;

    // Array of pointers to the accumulators
    v16acc64* accumulators[] = {
        &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
        &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
        &v16acc_partial8
    };

    // Determine the start and end of the loop based on the chunk index for weights
    const int input_channel_chunk_size = input_channels / input_split;
    const int start_ic = weight_index * input_channel_chunk_size;
    const int end_ic =  start_ic + input_channel_chunk_size;

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

        if(oc8==0 && end_ic== input_channels){ //if final set of input channels, scale the final output
            // Get cascade sum
            for (int pixel = 0; pixel < pixel_limit; pixel++) {
                int x = x_start + pixel;
                if (x < input_width) {
                    v16vec_cas[pixel] = lsrs(get_scd_v16acc64(),0,0); 
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
        if (weight_index != 1 && oc8==0) {  // Preload vector register with partial sum from previous iteration
            for (int pixel = 0; pixel < pixel_limit; pixel++) {
                int x = x_start + pixel;
                if (x < input_width) {
                    v16vec_partial[pixel] = lsrs(*accumulators[pixel],0,0); 
                }
            }
        }

        if (weight_index != 1) {  // Extract the partial sum
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
                v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum[pixel]);
            }
        }

        if (end_ic == input_channels) { // if final set of input channels, scale the final output
            for (int pixel = 0; pixel < pixel_limit; pixel++) {
                int x = x_start + pixel;
                if (x < input_width) {
                    cascade_sum=ext_elem(v16vec_cas[pixel], oc8);
                    // sum_srs[pixel] = ((cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs[pixel] = ((sum[pixel]+cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs[pixel] = (sum_srs[pixel] > MAX) ? MAX : (sum_srs[pixel] < -MAX) ? -MIN : sum_srs[pixel];

                    skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
                    skip_sum = sum_srs[pixel] + skip_temp;


                      skip_sum_srs_final =
                        (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
                    skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                            : (skip_sum_srs_final < -MAX)
                                                ? -MIN
                                                : skip_sum_srs_final; // clip
                    output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;
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
void conv2dk1_skip_ui8_i8_scalar_cascade_put(
    uint8_t *input0, int8_t *kernels, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels) {
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
        int sum_srs=0;
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
        v16vec_partial=upd_elem(v16vec_partial, value_index, sum);
        value_index++;
        if (value_index == MAX_VALUES) {
                // Transfer the values from vec to acc 
                v16acc_partial= lups(v16vec_partial,0);
                put_mcd(v16acc_partial); //push over cascade
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
    uint8_t *input0, int8_t *kernels, int8_t *output,int8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale, const int skip_scale) {
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
                v16acc_partial=get_scd_v16acc64(); // Get the accumulated values
                v16vec_partial= lsrs(v16acc_partial,0,0); // Convert accumulator to vector
                
        }

        // Extract the specific cascade sum for the current index
        int partial_sum=ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = half_input_channels/8; ic < input_channels / 8; ic++) {
          
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (half_input_channels / 8) * 64) + ((ic - half_input_channels / 8) * 64) + (ic8 * 8) + oc8];
            
            sum += val * k;
          }
        }
        
        if (value_index == MAX_VALUES) {
                value_index = 0;
        }
        // scale for convolution
        

        sum=sum+partial_sum;
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        //clip

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

#if defined (REGULAR) || (BN0) || (BN2) || (BN4) || (BN5) || (BN7) || (BN8) || (BN9)  || (BN11)
#ifdef UNSIGNED_SKIP
void conv2dk1_skip_ui8_ui8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, uint8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
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
        sum_srs = ((sum + (1 << (scaleT - 1)) - 1 + ((sum >> scaleT) & 1)) >> scaleT);
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs_out = (sum_srs > INT8_MAX) ? INT8_MAX : (sum_srs < INT8_MIN) ? INT8_MIN : sum_srs;
        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs_out + skip_temp;

        skip_sum_srs_final = ((skip_sum + (1 << (skip_scaleT - 1)) - 1 + ((skip_sum >> skip_scaleT) & 1)) >> skip_scaleT); 
        // skip_sum_srs_final = (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > INT8_MAX) ? INT8_MAX : (skip_sum_srs_final < INT8_MIN) ? INT8_MIN : skip_sum_srs_final;

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
void conv2dk1_skip_ui8_i8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, int8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
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
      sum_srs = ((sum + (1 << (scaleT - 1)) - 1 + ((sum >> scaleT) & 1)) >> scaleT);
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
      sum_srs_out = (sum_srs > INT8_MAX) ? INT8_MAX : (sum_srs < INT8_MIN) ? INT8_MIN : sum_srs;

        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs_out + skip_temp;

        skip_sum_srs_final = ((skip_sum + (1 << (skip_scaleT - 1)) - 1 + ((skip_sum >> skip_scaleT) & 1)) >> skip_scaleT); 
        // skip_sum_srs_final =(skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > INT8_MAX) ? INT8_MAX : (skip_sum_srs_final < INT8_MIN) ? INT8_MIN : skip_sum_srs_final;


        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

      }
    }
  }

  event1();
}

#endif
#endif // 
//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef BN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH

void bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip, 
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale, const int skip_scale,
                                              const int32_t input_split, const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {

    conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(input, kernels, output, skip,
                                                            input_width, input_channels,
                                                            output_channels, scale,skip_scale,
                                                            input_split,weight_index,x_start,oc) ;

                                              }
#endif
#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH

void bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(uint8_t *input, int8_t *kernels, int8_t *output, int8_t *skip, 
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale, const int skip_scale,
                                              const int32_t input_split, const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {

    conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get(input, kernels, output, skip,
                                                            input_width, input_channels,
                                                            output_channels, scale,skip_scale,
                                                            input_split,weight_index,x_start,oc) ;

                                              }
#endif

#ifdef BN13_2_PARTIAL_GET_I8_CAS_WIDTH
  void bn13_2_conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(uint8_t *input, int8_t *kernels, uint8_t *output,int8_t *skip, 
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale, const int skip_scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {
        conv2dk1_skip_ui8_i8_i8_scalar_partial_width_get(input,kernels, output,skip, 
                                              input_width,  input_channels,
                                              output_channels, scale, skip_scale,
                                              input_split, weight_index, x_start, oc); 
                                              }
#endif


#ifdef PUT
void conv2dk1_skip_ui8_i8_put(uint8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels) {
  conv2dk1_skip_ui8_i8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels);
}
#endif // PUT

#ifdef GET


void conv2dk1_skip_ui8_i8_i8_get(uint8_t *input0,int8_t *kernels,
                       int8_t *output, int8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_i8_i8_scalar_cascade_get(input0,  kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // GET

#ifdef BN0
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn0_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn0_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector
  #endif // Vector
#endif // BN0


#ifdef BN2
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn2_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn2_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector
  #endif // Vector
#endif // BN2

#ifdef BN4
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn4_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn4_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector
  #endif // Vector
#endif // BN4

#ifdef BN5
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn5_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn5_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector
  #endif // Vector
#endif // BN5

#ifdef BN7
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn7_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn7_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector


  #endif // Vector
#endif // BN7

#ifdef BN8
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn8_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn8_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector


  #endif // Vector
#endif // BN8

#ifdef BN11
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn11_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn11_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector


  #endif // Vector
#endif // BN9

#ifdef BN9
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void bn9_conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void bn9_conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector


  #endif // Vector
#endif // BN9

#ifdef REGULAR
  #ifdef SCALAR

    #ifdef UNSIGNED_SKIP

    void conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, uint8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #else

    void conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                          int8_t *output, int8_t *skip,
                          const int32_t input_width, const int32_t input_channels,
                          const int32_t output_channels, const int scale,
                          const int skip_scale) {
      conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                              input_channels, output_channels, scale, skip_scale);
    }

    #endif // UNSIGNED_SKIP

  #else // Vector


  #endif // Vector
#endif // REGULAR

} // extern "C"