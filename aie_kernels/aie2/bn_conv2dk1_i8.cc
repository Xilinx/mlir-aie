//===- conv2dk1.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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


const int32_t SMAX = 127;
const int32_t SMIN = 128;

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;




#if defined (BN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH) || (BN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH) 
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_ui8_ui8_scalar_input_split_partial_width_put(uint8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const  int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc) {
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

    v16acc64 acc_cas = undef_v16acc64();
    // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

    int pixel_limit = 7; // Process 8 pixels

    // Determine the start and end of the loop based on the chunk index for weights
    int input_channel_chunk_size = input_channels / input_split;
    int start_ic = weight_index * input_channel_chunk_size;
    int end_ic = start_ic + input_channel_chunk_size;

    // Preload vector register with partial sums from previous iteration
    v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
                               undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    if (weight_index != 0) {
        for (int pixel = 0; pixel < pixel_limit; pixel++) {
            int x = x_start + pixel;
            if (x < input_width) {
                v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
            }
        }
    }

    // Process each pixel across all output channels
    for (int pixel = 0; pixel < pixel_limit; pixel++) {
        // Loop over output channels (oc8)
        for (oc8 = 0; oc8 < 8; oc8++) {
          int last_sum =0;
          int current_sum = 0;
          int sum = 0;
            // Loop over input channels in chunks of 8
            for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
                for (ic8 = 0; ic8 < 8; ic8++) {
                        int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
                        int k = kernels[(0 * (input_channel_chunk_size / 8) * 64) +
                                        ((ic) * 64) + (ic8 * 8) + oc8];
                        current_sum += val * k;
                    
                }
            }
            if (weight_index != 0) {  // Preload with partial sum from previous iteration
                last_sum = ext_elem(v16vec_partial[pixel], oc8);
            }
            sum = current_sum + last_sum;
            v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
            // Update accumulators
            
            // Store the sum for the next iteration
            if (weight_index != (input_split / 2 - 1)) {
               *accumulators[pixel] = lups(v16vec_partial[pixel],0);
            }
        }
      if (weight_index == (input_split / 2 - 1)) {
                acc_cas= lups(v16vec_partial[pixel],0);
                put_mcd(acc_cas); //push over cascade
      }
    }

    event1();
}
#endif


#if defined (PARTIAL_PUT_I8_CAS_WIDTH_NEW)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_i8_ui8_scalar_partial_width_put_new(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const  int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc) {
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

    v16acc64 acc_cas = undef_v16acc64();
    // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

    int pixel_limit = 7; // Process 8 pixels

    // Determine the start and end of the loop based on the chunk index for weights
    int input_channel_chunk_size = input_channels / input_split;
    int start_ic = weight_index * input_channel_chunk_size;
    int end_ic = start_ic + input_channel_chunk_size;

    // Preload vector register with partial sums from previous iteration
    v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
                               undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    if (weight_index != 0) {
        for (int pixel = 0; pixel < pixel_limit; pixel++) {
            int x = x_start + pixel;
            if (x < input_width) {
                v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
            }
        }
    }

    // Process each pixel across all output channels
    for (int pixel = 0; pixel < pixel_limit; pixel++) {
        // Loop over output channels (oc8)
        for (oc8 = 0; oc8 < 8; oc8++) {
          int last_sum =0;
          int current_sum = 0;
          int sum = 0;
            // Loop over input channels in chunks of 8
            for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
                for (ic8 = 0; ic8 < 8; ic8++) {
                        int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
                        int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) +
                                        ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
                        current_sum += val * k;
                    
                }
            }
            if (weight_index != 0) {  // Preload with partial sum from previous iteration
                last_sum = ext_elem(v16vec_partial[pixel], oc8);
            }
            sum = current_sum + last_sum;
            v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
            // Update accumulators
            
            // Store the sum for the next iteration
            if (weight_index != (input_split / 2 - 1)) {
               *accumulators[pixel] = lups(v16vec_partial[pixel],0);
            }
        }
      if (weight_index == (input_split / 2 - 1)) {
                acc_cas= lups(v16vec_partial[pixel],0);
                put_mcd(acc_cas); //push over cascade
      }
    }

    event1();
}
#endif


#if defined (PARTIAL_PUT_I8_CAS_WIDTH) || (BN13_1_PARTIAL_PUT_I8_CAS_WIDTH) || (BN14_1_PARTIAL_PUT_I8_CAS_WIDTH)
// 8 Pixels Width Processing Approach: Processes 8 spatial pixels (x_start to x_start + 8) simultaneously within each output channel (oc8 iteration).
void conv2dk1_i8_ui8_scalar_partial_width_put(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const  int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc) {
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

    v16acc64 acc_cas = undef_v16acc64();
    // static v16acc64 v16acc_partial[8]; // Using an array for accumulators

    int pixel_limit = 7; // Process 8 pixels

    // Determine the start and end of the loop based on the chunk index for weights
    int input_channel_chunk_size = input_channels / input_split;
    int start_ic = weight_index * input_channel_chunk_size;
    int end_ic = start_ic + input_channel_chunk_size;

    // Preload vector register with partial sums from previous iteration
    v16int32 v16vec_partial[8] = {undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32(),
                               undef_v16int32(), undef_v16int32(), undef_v16int32(), undef_v16int32()};
    if (weight_index != 0) {
        for (int pixel = 0; pixel < pixel_limit; pixel++) {
            int x = x_start + pixel;
            if (x < input_width) {
                v16vec_partial[pixel] = lsrs(*accumulators[pixel], 0, 0);
            }
        }
    }

    // Process each pixel across all output channels
    for (int pixel = 0; pixel < pixel_limit; pixel++) {
        // Loop over output channels (oc8)
        for (oc8 = 0; oc8 < 8; oc8++) {
          int last_sum =0;
          int current_sum = 0;
          int sum = 0;
            // Loop over input channels in chunks of 8
            for (ic = start_ic / 8; ic < end_ic / 8; ic++) {
                for (ic8 = 0; ic8 < 8; ic8++) {
                        int val = input[(ic * input_width * 8) + (pixel * 8) + ic8];
                        int k = kernels[(0 * (input_channel_chunk_size / 8) * 64) +
                                        ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
                        current_sum += val * k;
                    
                }
            }
            if (weight_index != 0) {  // Preload with partial sum from previous iteration
                last_sum = ext_elem(v16vec_partial[pixel], oc8);
            }
            sum = current_sum + last_sum;
            v16vec_partial[pixel] = upd_elem(v16vec_partial[pixel], oc8, sum);
            // Update accumulators
            
            // Store the sum for the next iteration
            if (weight_index != (input_split / 2 - 1)) {
               *accumulators[pixel] = lups(v16vec_partial[pixel],0);
            }
        }
      if (weight_index == (input_split / 2 - 1)) {
                acc_cas= lups(v16vec_partial[pixel],0);
                put_mcd(acc_cas); //push over cascade
      }
    }

    event1();
}
#endif

#ifdef PARTIAL_PUT_I8_CAS
// Output Channel First Approach: Iterates over each output channel (oc) first and then processes all pixels (x) within that output channel iteration.
void conv2dk1_i8_scalar_partial_put(int8_t *input, int8_t *kernels,
                                        const int32_t input_width, const int32_t input_channels,
                                        const int32_t output_channels, int32_t input_split,
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
  v16acc64* accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
      &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
      &v16acc_partial8
  };


  // Determine the start and end of the loop based on the chunk index for weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;
  for (oc = 0; oc < output_channels / 8; oc++) {
  // for (x = 0; x < input_width; x++) { // col of output image
    v16acc64& accumulator = *accumulators[oc%9];
    v16int32 v16vec_partial = lsrs(accumulator,0,0); 
    int value_index = 0;
    v16acc64 acc_cas = undef_v16acc64();

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
        last_sum=ext_elem(v16vec_partial, oc8);
      }

      sum=current_sum+last_sum;

      // Transfer scalar sum to vector
      v16vec_partial=upd_elem(v16vec_partial, oc8, sum); 


      if(oc8==7 && end_ic==input_channels/2){ //if final set of input channels, scale the final output
            acc_cas= lups(v16vec_partial,0);
            put_mcd(acc_cas); //push over cascade
      }
      
      
      if (oc8 == 7 && end_ic != input_channels/2) { //end of vectorization
            // // Transfer the values from vec to acc 
          accumulator= lups(v16vec_partial,0);
      }
    } 
  }
  // }
  event1();
}

#endif

//*****************************************************************************
// conv2d 1x1_PUT - scalar
// act: int8, wts: int8, cascade: uint8 (CHECK)
//*****************************************************************************
#ifdef PUT_I8_CAS_REPEAT
void conv2dk1_i8_scalar_cascade_put_repeat(
    int8_t *input0, int8_t *kernels, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split,const int32_t weight_index,int oc) {
  event0();

  int x, ic, ic2, oc8, ic8, ic8b;

  v16int32 v16vec_partial = undef_v16int32();
  // v16acc64 v16acc_partial = undef_v16acc64();
  v16acc64 accumulators[12];
  for (int i = 0; i < 12; i++) {
    accumulators[i] = undef_v16acc64();
  }
  int accumulator_index = 0;
  int value_index = 0;

  // Calculate half the input channels
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;

  // for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
        if(weight_index==0)
          sum[x] = 0;

        for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            
            sum[x] += val * k;
          }
        }
        
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        // sum_srs = (sum_srs > MAX)    ? MAX
        //           : (sum_srs < -MIN) ? -MIN
        //                              : sum_srs; // clip
        v16vec_partial=upd_elem(v16vec_partial, value_index, sum[x]);
        value_index++;
        if (value_index == MAX_VALUES) {
                // Transfer the values from vec to acc 
                accumulators[accumulator_index] = lups(v16vec_partial,0);
                put_mcd(accumulators[accumulator_index]); // Push over cascade
                // Reset the index
                value_index = 0;

                // Move to the next accumulator register
          accumulator_index++;
          if (accumulator_index == 12) {
            // If all accumulator registers are used, reset the index
            accumulator_index = 0;
          }
        }
      }
    }
  // }

  event1();
}
#endif




#ifdef PUT_I8_CAS
void conv2dk1_i8_scalar_cascade_put(
    int8_t *input0, int8_t *kernels, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split,const int32_t weight_index) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;

  // Calculate half the input channels
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
        if(weight_index==0)
          sum[x] = 0;

        for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            
            sum[x] += val * k;
          }
        }
        
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        // sum_srs = (sum_srs > MAX)    ? MAX
        //           : (sum_srs < -MIN) ? -MIN
        //                              : sum_srs; // clip
        v16vec_partial=upd_elem(v16vec_partial, value_index, sum[x]);
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


#ifdef PUT_UI8_CAS
void conv2dk1_ui8_scalar_cascade_put(
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
void conv2dk1_i8_ui8_scalar_cascade_get(
    int8_t *input0, int8_t *kernels, uint8_t *output,
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;
  
  const int scaleT = scale;
  const int half_input_channels = input_channels / 2;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;

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
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (((sum) + (1 << (scaleT - 1)) - 1 + (((sum) >> scaleT) & 1)) >> scaleT);
        sum_srs = (sum_srs > UMAX)    ? UMAX
                  : (sum_srs < 0) ? 0
                                     : sum_srs; // clip
        //clip

        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif


#ifdef SCALAR
//*****************************************************************************
// conv2d 1x1 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, int8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
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

        // sum_srs=sum>>scale;
        // sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = ((sum + (1 << (scale - 1)) - 1 + ((sum >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > SMAX) ? SMAX : (sum_srs < -SMIN) ? -SMIN : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }
event1();
}

#endif

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef BN0
    void bn0_conv2dk1_ui8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN0

#ifdef BN1
    void bn1_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN1

#ifdef BN3
    void bn3_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN5
#ifdef BN5
    void bn5_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN5
#ifdef BN6
    void bn6_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN6

#ifdef BN8
    void bn8_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN8

// #ifdef BN7
//     void bn7_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
//                       const int32_t input_width, const int32_t input_channels,
//                       const int32_t output_channels, const int scale) {
//       conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
//                           output_channels, scale);
//     }

// #endif // BN6

#ifdef BN10
    void bn10_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }
#endif //BN12

#ifdef BN12
    void bn12_conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN10

 #ifdef BN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH
void bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put(uint8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {
        conv2dk1_ui8_ui8_scalar_input_split_partial_width_put(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif

  #ifdef BN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH
void bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put(uint8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {
        conv2dk1_ui8_ui8_scalar_input_split_partial_width_put(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif

  #ifdef BN13_2_PARTIAL_GET_I8_I8_CAS_WIDTH

void bn13_2_conv2dk1_i8_i8_partial_width_get(int8_t *input, int8_t *kernels, int8_t *output,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int scale,
                                              int32_t input_split, int32_t weight_index, int32_t x_start, int32_t oc) 
                                              {

    conv2dk1_i8_i8_scalar_partial_width_get(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,x_start,oc) ;

                                              }
#endif


#ifdef BN13_1_PARTIAL_PUT_I8_CAS_WIDTH
void bn13_1_conv2dk1_i8_ui8_partial_width_put(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {
        conv2dk1_i8_ui8_scalar_partial_width_put(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif

#ifdef BN14_1_PARTIAL_PUT_I8_CAS_WIDTH
void bn14_1_conv2dk1_i8_ui8_partial_width_put(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, const int32_t input_split,
                                              const int32_t weight_index, const int32_t x_start, const int32_t oc)
                                              {
        conv2dk1_i8_ui8_scalar_partial_width_put(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif

#ifdef PARTIAL_PUT_I8_CAS_WIDTH
void conv2dk1_i8_ui8_partial_width_put(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, int32_t input_split,
                                              int32_t weight_index, int32_t x_start, int32_t oc)
                                              {
        conv2dk1_i8_ui8_scalar_partial_width_put(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif
#ifdef PARTIAL_PUT_I8_CAS_WIDTH_NEW
void conv2dk1_i8_ui8_partial_width_put_new(int8_t *input, int8_t *kernels,
                                              const int32_t input_width, const int32_t input_channels,
                                              const int32_t output_channels, int32_t input_split,
                                              int32_t weight_index, int32_t x_start, int32_t oc)
                                              {
        conv2dk1_i8_ui8_scalar_partial_width_put_new(input, kernels,
                                          input_width, input_channels,
                                          output_channels,  input_split,
                                          weight_index,  x_start,  oc);
                                              }

#endif
#ifdef PARTIAL_PUT_I8_CAS
void conv2dk1_i8_partial_put(int8_t *input, int8_t *kernels,
                                        const int32_t input_width, const int32_t input_channels,
                                        const int32_t output_channels, 
                                        int32_t input_split, int32_t weight_index, int32_t x) {
  conv2dk1_i8_scalar_partial_put(input, kernels, 
                          input_width, input_channels,output_channels, 
                          input_split,weight_index,x);
}
#endif // PUT



#ifdef PUT_I8_CAS_REPEAT
void conv2dk1_i8_put_repeat(int8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels,const int32_t input_split,const int32_t weight_index,int oc) {
  conv2dk1_i8_scalar_cascade_put_repeat(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels, input_split,weight_index,oc);
}
#endif // PUT

#ifdef PUT_I8_CAS
void conv2dk1_i8_put(int8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels,const int32_t input_split,const int32_t weight_index) {
  conv2dk1_i8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels, input_split,weight_index);
}
#endif // PUT

#ifdef PUT_UI8_CAS
void conv2dk1_ui8_put(uint8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels) {
  conv2dk1_ui8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels);
}
#endif // PUT


#ifdef GET
void conv2dk1_i8_ui8_get(int8_t *input0,int8_t *kernels,
                       uint8_t *output,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale
                       ) {
  conv2dk1_i8_ui8_scalar_cascade_get(input0,  kernels, output, input_width,
                           input_channels, output_channels, scale);
}

#endif // GET

#ifdef REGULAR
void conv2dk1_ui8_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}
#endif // REGULAR
} // extern "C"