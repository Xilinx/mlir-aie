//===- conv2dk1.cc -------------------------------------------------*- C++
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

#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1


const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;




#if defined (BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_ui8_scalar_input_split_partial_width_get(uint8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              const int32_t input_split, const int32_t weight_index, const int32_t x_start, const int32_t oc) {
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
                    sum = ext_elem(v16vec_partial[pixel], oc8);
                    cascade_sum=ext_elem(v16vec_cas[pixel], oc8);
                    sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs= (sum_srs > UMAX) ? UMAX : (sum_srs< 0) ? 0 : sum_srs;
                    output[(oc * input_width * 8) + (pixel * 8) + oc8] = sum_srs;
            }
        }       
    } 

    event1();
}
#endif



#if defined (PARTIAL_GET_I8_CAS_WIDTH_NEW)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_i8_ui8_scalar_partial_width_get_new(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t output_split,int32_t weight_index, int32_t x_start, int32_t oc) {
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
    int start_ic = 1 * input_channel_chunk_size;
    int end_ic =  start_ic + input_channel_chunk_size;
    int pixel = 0;
    int oc_offset=0;
    // Preload vector register with partial sums from previous iteration
    // v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
    //                            undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    // v16int32 v16vec_cas[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
    //                            undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    v16acc64 acc_cas = undef_v16acc64();
    v16int32 v16vec_partial[8] = {}; 
    v16int32 v16vec_cas[8] = {}; 

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
                            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                                            ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
                            current_sum+= val * k;
                    }
                }
                // Transfer scalar sum to vector
                sum= current_sum;
                v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
            }

            // if (end_ic == input_channels) {
                // acc_cas=get_scd_v16acc64();
                // int scale_new=8;
                v16vec_cas[pixel] = lsrs(get_scd_v16acc64(),0, 0);
                for (oc8 = 0; oc8 < 8; oc8++) {
                    int sum = 0;
                    int sum_srs = 0;
                    int cascade_sum = 0;
                    sum = ext_elem(v16vec_partial[pixel], oc8);
                    cascade_sum=ext_elem(v16vec_cas[pixel], oc8);
                    // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs = (((cascade_sum) + (1 << (scale - 1)) - 1 + (((cascade_sum) >> scale) & 1)) >> scale);
                    sum_srs= (sum_srs > UMAX) ? UMAX : (sum_srs< 0) ? 0 : sum_srs;
                    if(weight_index!=0)
                        // oc_offset=oc8+weight_index*8;
                        
                        if(oc==0 && output_split==4 )
                            oc_offset=oc+(weight_index);
                        else
                            oc_offset=oc+output_split*(weight_index);
                    else
                        oc_offset=oc;
                    output[(oc_offset * input_width * 8) + (pixel * 8) + (oc8)] = sum_srs;
            }
        // }       
    } 

    event1();
}
#endif


#if defined (PARTIAL_GET_I8_CAS_WIDTH) || (BN13_2_PARTIAL_GET_I8_CAS_WIDTH) || (BN13_1_PARTIAL_GET_I8_CAS_WIDTH) || (BN14_1_PARTIAL_GET_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).

void conv2dk1_i8_ui8_scalar_partial_width_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) {
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

    if (weight_index > input_split/2 ) {  // Preload vector register with partial sum from previous iteration. If weight is only 1 then we don't have partial sum anyway
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
                                            ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
                            current_sum+= val * k;
                    }
                }
                // Extract the partial sum if applicable
                if (weight_index > input_split / 2) {
                        last_sum = ext_elem(v16vec_partial[pixel], oc8);
                }
                // Transfer scalar sum to vector
                sum= current_sum + last_sum;
                v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
                if (input_split != (weight_index + 1)) {
                    *accumulators[pixel] = lups(v16vec_partial[pixel],0);
                }
            }

            if (end_ic == input_channels) {
                // acc_cas=get_scd_v16acc64();
                // int scale_new=8;
                v16vec_cas[pixel] = lsrs(get_scd_v16acc64(),0, 0);
                for (oc8 = 0; oc8 < 8; oc8++) {
                    int sum = 0;
                    int sum_srs = 0;
                    int cascade_sum = 0;
                    sum = ext_elem(v16vec_partial[pixel], oc8);
                    cascade_sum=ext_elem(v16vec_cas[pixel], oc8);
                    // sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
                    sum_srs = (((sum+cascade_sum) + (1 << (scale - 1)) - 1 + (((sum+cascade_sum) >> scale) & 1)) >> scale);
                    sum_srs= (sum_srs > UMAX) ? UMAX : (sum_srs< 0) ? 0 : sum_srs;
                    output[(oc * input_width * 8) + (pixel * 8) + oc8] = sum_srs;
            }
        }       
    } 

    event1();
}
#endif

#ifdef PARTIAL_WIDTH
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_ui8_scalar_partial_width(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale,
                        int32_t input_split,int32_t weight_index,int32_t x_start, int32_t oc ) {
  
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
        if (weight_index != 0 && oc8==0) {  // Preload vector register with partial sum from previous iteration
            for (int pixel = 0; pixel < 8; pixel++) {
                int x = x_start + pixel;
                if (x < input_width) {
                    v16vec_partial[pixel] = lsrs(*accumulators[pixel],0,0); 
                }
            }
        }

        if (weight_index != 0) {  // Extract the partial sum
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
                v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum[pixel]);
            }
        }

        if (end_ic == input_channels) { // if final set of input channels, scale the final output
            for (int pixel = 0; pixel < 8; pixel++) {
                int x = x_start + pixel;
                if (x < input_width) {
                    sum_srs[pixel] = (sum[pixel] + (1 << (scale - 1))) >> scale;
                    sum_srs[pixel] = (sum_srs[pixel] > UMAX) ? UMAX : (sum_srs[pixel] < 0) ? 0 : sum_srs[pixel];
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
// Output Channel First Approach: Iterates over each output channel (oc) first and then processes all pixels (x) within that output channel iteration.
void conv2dk1_i8_ui8_scalar_partial_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                        const int32_t input_width, const int32_t input_channels,
                                        const int32_t output_channels, const int scale,
                                        int32_t input_split, int32_t weight_index, int32_t x) {
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
  v16acc64* accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
      &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
      &v16acc_partial8
  };

  // static v16acc64 v16acc_partial;

  // Determine the start and end of the loop based on the chunk index for weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;
  for (oc = 0; oc < output_channels / 8; oc++) {
  // for (x = 0; x < input_width; x++) { // col of output image
    v16acc64& accumulator = *accumulators[oc%9];
    v16int32 v16vec_partial = lsrs(accumulator,0,0); 
    int value_index = 0;
    int cascade_sum = 0;
    v16acc64 acc_cas = undef_v16acc64(); // Get the accumulated values
    v16int32 vec_cas= undef_v16int32(); // Convert accumulator to vector
   
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int sum_srs = 0;
      int last_sum = 0;

      if(oc8==0 && end_ic== input_channels){ //if final set of input channels, scale the final output
            // Get cascade sum
            acc_cas=get_scd_v16acc64(); // Get the accumulated values
            vec_cas= lsrs(acc_cas,0,0); // Convert accumulator to vector
      }

      
      //Current iteration: go over all the input channels
      for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                    ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
            current_sum += val * k;
          }
      }
    
      if (weight_index != 1){  // Extract the partial sum 
        last_sum=ext_elem(v16vec_partial, oc8);
      }

     
      sum=current_sum+last_sum;

      // Transfer scalar sum to vector
      v16vec_partial=upd_elem(v16vec_partial, oc8, sum); 

      if(end_ic == input_channels){ //if final set of input channels, scale the final output
            cascade_sum=ext_elem(vec_cas, oc8);
            sum_srs = ((sum+cascade_sum) + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
            // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
            output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }

      
      if (oc8 == 7) { //end of vectorization
            // // Transfer the values from vec to acc 
          accumulator= lups(v16vec_partial,0);
      }
    } 
  }
  // }
  event1();
}

#endif

#ifdef PARTIAL
// Output Channel First Approach: Iterates over each output channel (oc) first and then processes all pixels (x) within that output channel iteration.
//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_ui8_scalar_partial(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale,
                        int32_t input_split,int32_t weight_index,int32_t x ) {
  
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
  v16acc64* accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
      &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
      &v16acc_partial8
  };

  // static v16acc64 v16acc_partial;

  // Determine the start and end of the loop based on the chunk index for weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;
  for (oc = 0; oc < output_channels / 8; oc++) {
  // for (x = 0; x < input_width; x++) { // col of output image
    v16acc64& accumulator = *accumulators[oc%9];
    v16int32 v16vec_partial = lsrs(accumulator,0,0); 
    int value_index = 0;

    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int sum_srs = 0;
      int last_sum = 0;


      //Current iteration: go over all the input channels
      for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                    ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
            current_sum += val * k;
          }
      }
    
      if (weight_index != 0){  // Extract the partial sum 
        last_sum=ext_elem(v16vec_partial, value_index);
      }

      sum=current_sum+last_sum;

      // Transfer scalar sum to vector
      v16vec_partial=upd_elem(v16vec_partial, value_index, sum); 
      value_index++; 

      if(end_ic == input_channels){ //if final set of input channels, scale the final output
            // Transfer the values from acc to vect 
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
            // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
            output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
      
      
      if (oc8 == 7) { //end of vectorization
            // // Transfer the values from vec to acc 
          accumulator= lups(v16vec_partial,0);
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
    int8_t *input0, int8_t *kernels, uint8_t *output,
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split,const int32_t weight_index,
    const int scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;
  
  const int scaleT = scale;
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic = input_channels/2 + weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
       if(weight_index==0)
          sum[x] = 0;
        int sum_srs = 0;

        // Extract cascade sum values when starting a new block
        if (value_index == 0) {
                v16acc_partial=get_scd_v16acc64(); // Get the accumulated values
                v16vec_partial= lsrs(v16acc_partial,0,0); // Convert accumulator to vector
                
        }

        // Extract the specific cascade sum for the current index
        int partial_sum=ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) 
                            + ((ic - input_channel_chunk_size / 8) * 64) 
                            + (ic8 * 8) + oc8];
            
            sum[x] += val * k;
          }
        }
        
        if (value_index == MAX_VALUES) {
                value_index = 0;
        }
        // scale for convolution
        sum[x]=sum[x]+partial_sum;
        // sum=partial_sum;
        if(end_ic == input_channels){
          sum_srs = (sum[x] + (1 << (scaleT - 1))) >> scaleT;
          sum_srs = (sum_srs > UMAX)    ? UMAX
                    : (sum_srs < 0) ? 0
                                      : sum_srs; // clip
          //clip

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
//                         const int32_t input_width, const int32_t input_channels,
//                         const int32_t output_channels, const int scale) {
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
//                         const int32_t input_width, const int32_t input_channels,
//                         const int32_t output_channels, const int scale) {
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
//                         const int32_t input_width, const int32_t input_channels,
//                         const int32_t output_channels, const int scale) {
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
//         sum_srs = ((sum + (1 << (applied_scale - 1)) - 1 + ((sum >> applied_scale) & 1)) >> applied_scale);
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

#if defined (BN1)||(BN2) ||(BN3) ||(BN4) || (BN5) || (BN6) || (BN7) || (BN8) || (BN9) || (BN10) || (BN11)|| (BN12) || (BN13) || (BN14) ||  (REGULAR)
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
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
        sum_srs = ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        
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

// #if defined (BN1) ||(BN2) ||(BN3) ||(BN4) || (BN5) || (BN6) || (BN7) || (BN8) || (BN9) || (BN10) || (BN11) || (BN12) || (BN13) || (BN14) ||  (REGULAR)
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
//         sum_srs = ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
//         sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
//         // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
//         output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
//       }
//     }
//   }

//   event1();
// }

// #endif // UINT8_ACT
// #endif


//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef BN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_UI8_CAS_WIDTH

void bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_get(uint8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              const int32_t input_split, const int32_t weight_index, const int32_t x_start,const int32_t oc) 
                                              {

    conv2dk1_ui8_ui8_scalar_input_split_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

                                              }
#endif

#ifdef BN13_2_PARTIAL_GET_I8_CAS_WIDTH

void bn13_2_conv2dk1_i8_ui8_partial_width_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_ui8_scalar_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

                                              }
#endif

#ifdef BN13_1_PARTIAL_GET_I8_CAS_WIDTH

void bn13_1_conv2dk1_i8_ui8_partial_width_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_ui8_scalar_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

                                              }
#endif

#ifdef BN14_1_PARTIAL_GET_I8_CAS_WIDTH

void bn14_1_conv2dk1_i8_ui8_partial_width_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_ui8_scalar_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

                                              }
#endif
#ifdef PARTIAL_GET_I8_CAS_WIDTH_NEW

void conv2dk1_i8_ui8_partial_width_get_new(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split,int32_t output_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_ui8_scalar_partial_width_get_new(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,output_split,weight_index,x_start,oc) ;

                                              }
#endif
  
#ifdef PARTIAL_GET_I8_CAS_WIDTH

void conv2dk1_i8_ui8_partial_width_get(int8_t *input, int8_t *kernels, uint8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_ui8_scalar_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

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
//                       const int32_t input_width, const int32_t input_channels,
//                       const int32_t output_channels, const int scale) {
//       conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
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
//                       const int32_t input_width, const int32_t input_channels,
//                       const int32_t output_channels, const int scale) {
//       conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
//                           output_channels, scale);
//     }

//     #endif // UINT8_ACT

// #endif // Vector
  
  
// #ifdef BN11

//       #ifdef INT8_ACT

//       void bn11_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
//                       const int32_t input_width, const int32_t input_channels,
//                       const int32_t output_channels, const int scale) {
//         conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
//                           output_channels, scale);
//       }

//       #else // UINT8_ACT

//       void bn11_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
//                         const int32_t input_width, const int32_t input_channels,
//                         const int32_t output_channels, const int scale) {
//         conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
//                             output_channels, scale);
//       }

//       #endif // UINT8_ACT



// #endif
 #ifdef BN1
void bn1_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN2
void bn2_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN3
void bn3_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif
 #ifdef BN4
void bn4_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN5
void bn5_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif
 #ifdef BN6
void bn6_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN7
void bn7_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN8
void bn8_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN9
void bn9_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN10
void bn10_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN11
void bn11_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif

 #ifdef BN12
void bn12_conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

                 }

#endif
 #ifdef REGULAR
void conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

}
#endif

} // extern "C"