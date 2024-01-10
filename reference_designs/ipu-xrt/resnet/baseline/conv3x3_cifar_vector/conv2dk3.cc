//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

//#define __AIENGINE__ 1
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {
enum region{top,middle,bottom};


    void conv2dk3(int8_t *line0, int8_t *line1,int8_t *line2,int8_t *wts, uint8_t *output,  const int32_t  input_width,  const int32_t  input_channels, const int32_t  output_channels,
                        const int32_t  kernel_width,  const int32_t  kernel_height,  const int32_t  check,const int scale, const int channel_offset)
     {
        event0();

        // Compute 
        using MMUL4x8x8 = aie::mmul<4,8,8, int8, int8>;
        ::aie::set_saturation(aie::saturation_mode::saturate); // Needed to saturate properly to uint8
        ::aie::set_rounding(aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

        constexpr unsigned VecFactor = 16;

        // const int scale = 11;

        // basic MMUL intrinisic needed is k x ic x oc
        // k is number of inputs processed at a time
        // So if ic=8, oc=4, then k=8 and we use 8x8x4
        const unsigned k = 256/(input_channels*output_channels); // 8 inputs per vector output

        aie::vector<int8,32> zero32 = aie::zeros<int8,32>();

        aie::vector<int8, 64> in_a;
        aie::vector<int8, 64> prev_a[3], tmp_a;
        aie::vector<int8, 64> in_b;
        int8_t * restrict line[3];
        line[0] = line0;
        line[1] = line1;
        line[2] = line2;

        int8_t * restrict wtsLine[3];
        // oc,ic,ky,kx,ic8,oc8
        wtsLine[0] = wts + (channel_offset/8)*(input_channels/8)*kernel_height*kernel_width*64;
        wtsLine[1] = wts + (channel_offset/8)*(input_channels/8)*kernel_height*kernel_width*64 + kernel_width*64;   // next kernel line is always 8*8 away
        wtsLine[2] = wts + (channel_offset/8)*(input_channels/8)*kernel_height*kernel_width*64 + 2*kernel_width*64; // next kernel line is always 8*8 away

        aie::vector<int8, 32> tmp_a1, tmp_a2;

        MMUL4x8x8 acc_tmp[8];

        // Zero accumulators used for storing partial results
        // for(int x=0; x<input_width/4-2; x++) {
        // for(int x=0; x<(iw/4)-2; x++) {
        for(int x=0; x<8; x++) {
            acc_tmp[x] = aie::zeros<acc32,32>();   
        }

        // TODO temporary workaround. When assigned to input_width, the results are wrong. ???
        const int iw = 32;
        // const int32_t iw = input_width;

        // const int iw_32 = ((input_width/4)-2)/8;
        // const int iw_32 = ((iw/4)-2)/8;
        // const int iw_32 = ((32/4)-2)/8;
        const int iw_32 = 0;

        // const int iw_32_rem = ((input_width/4)-2) % 8;
        // const int iw_32_rem = ((iw/4)-2) % 8;
        // const int iw_32_rem = ((32/4)-2) % 8;
        const int iw_32_rem = 6;

        output += (channel_offset*iw); // channel_offset/8*iw*8

        int kernel_height_start;
        int kernel_height_end;

        // int kernel_height_start, kernel_height_end;
    #ifdef BORDER_REPLICATE 
        kernel_height_start = 0;
        kernel_height_end   = kernel_height;
        // constexpr int kernel_height_start = 0;
        // constexpr int kernel_height_end   = kernel_height;
    #else
        // constexpr int kernel_height_start = 0;
        // constexpr int kernel_height_end   = kernel_height-1;

        // if(check == top)
        //     idx_adj = 1;


        switch(check) {
            case top:
                kernel_height_start = 1;
                kernel_height_end   = kernel_height;
                break;
            case middle:
                kernel_height_start = 0;
                kernel_height_end   = kernel_height;
                break;
            case bottom:
                kernel_height_start = 0;
                kernel_height_end   = kernel_height-1;
                break;
        }   
    #endif

        // Left pattern
        {
            // in_b = aie::load_v<64>(wtsLine[kernel_height_start]); wtsLine[kernel_height_start] +=64;       // wts ic0..7(oc0..7)

            MMUL4x8x8 acc1 = aie::zeros<acc32,32>();

            for(int oc=0; oc<(output_channels/8); oc++) {
                for(int ic=0; ic<(input_channels/8); ic++) {
                    for(int i=kernel_height_start; i<kernel_height_end; i++)
                        chess_prepare_for_pipelining chess_loop_range(2,)    
                        // chess_unroll_loop()
                    {
                        tmp_a1    = aie::load_v<32>(line[i]); line[i]+=32; // act 0..7(ic0..7) -- but only use 0..3
                        tmp_a2    = aie::load_v<32>(line[i]);
                        in_a      = aie::concat(tmp_a1, tmp_a2);

    #ifdef BORDER_REPLICATE 
                        tmp_a1    = aie::shuffle_up(tmp_a1, 24);
                        tmp_a.insert<32>(1, tmp_a1);
    #else
                        tmp_a     = aie::zeros<int8, 64>();
    #endif
                        in_a      = aie::shuffle_up_fill(in_a, tmp_a, 8);
                        // prev_a[i] = aie::shuffle_up(in_a, 24); // Shift right (4-1)*8
                        prev_a[i] = in_a;
                        prev_a[i] = aie::shuffle_up(prev_a[i], 24); // Shift right (4-1)*8

                        for(int j=0; j<kernel_width; j++)
                        // chess_unroll_loop()
                        {
                            in_b  = aie::load_v<64>(wtsLine[i]); wtsLine[i] +=64; // wts ic0..7(oc0..7)
                            acc1.mac(in_a.extract<32>(0), in_b);
                            // Shift input A by 1 row (1x8) which is by 1 (the 8 is the ic=8)
                            in_a = aie::shuffle_down(in_a, 8);
                        }
                        wtsLine[i] -= (kernel_width*64); // Reset previous wegiht pointer to beginning
                    }

                    // Reset weights and input pointer for next ic/8
                    for(int i=kernel_height_start; i<kernel_height_end; i++) {
                        wtsLine[i] += kernel_width*kernel_height*64; // kernel_width*kernel_height*8*8
                        line[i]    += (iw-4)*8; // (iw-4)*8, length of act minus 1 vlds to shift back
                    }
                } // for(int ic=0; ic<(input_channels/8); ic++) {

                aie::vector<uint8,32> o1 = acc1.to_vector<uint8>(scale);
                aie::store_v(output, o1); 
                output += iw*8; // Shift to next oc/8 offset for left side

                acc1 = aie::zeros<acc32,32>();            
                
                for(int i=kernel_height_start; i<kernel_height_end; i++) {
                    line[i]    -= (input_channels/8)*(iw*8); // shift back to beginning of this section
                }

            } // for(int oc=0; oc<(output_channels/8); oc++) {

            // Reset output, wts, line pointers
            output -= (output_channels/8)*(iw*8)-32;
            for(int i=kernel_height_start; i<kernel_height_end; i++) {
                wtsLine[i] -= (output_channels/8)*(input_channels/8)*kernel_width*kernel_height*64; // kernel_width*kernel_height*8*8
                // line[i]    -= (output_channels/8)*(input_channels/8)*(iw*8)-32; // 
                line[i]    += 32;
            }
        }


        // H,W,C8
        // Middle pattern

        if(iw_32 > 0) {

        // Main loop for when input_width/4-2 > 8
        for(int oc=0; oc<(output_channels/8); oc++) {
            for(int iw_32c=0; iw_32c<iw_32; iw_32c++) {
                for(int ic=0; ic<(input_channels/8); ic++) {
                    for(int i=kernel_height_start; i<kernel_height_end; i++) { // 1 to 3



                        for(int j=0; j<kernel_width;j++) {
                            aie::vector<int8, 64> wtsVec = aie::load_v<64>(wtsLine[i]); wtsLine[i] += 64;

                            // auto prev = prev_a[i].extract<32>(1);                  // prev = x0..x3(ci0..ci7)
                            auto prev = aie::load_v<32>((line[i]-32));
                            auto curr = aie::load_v<32>((line[i])); line[i]+=32;
                            auto next = aie::load_v<32>((line[i])); line[i]+=32;

                            for(int x=0; x<8; x++) 
                                // chess_unroll_loop()
                            {
                                auto tmp1   = aie::concat(curr,next);
                                auto tprev  = aie::concat(zero32, prev);
                                auto tmp2   = aie::shuffle_up_fill(tmp1, tprev, 8); // curr      = x3..x6(ci0..ci7)
                                auto tmp3   = aie::shuffle_down(tmp2, j*8);             // curr      = x4..x7(ci0..ci7) to x5..x8(ci0..ci7)ss

                                prev = curr;
                                curr = next;
                                next = aie::load_v<32>(line[i]); line[i]+=32; // next_prev = x4..x7(ci0..ci7)

                                acc_tmp[x].mac(tmp3.extract<32>(0), wtsVec);
                            } // for(int x=0; x<8; x++) 
                            line[i] -= 320; // (8+2)*32, Reset line buffer ptr to beginning of line (after first 4)
                        } // for(int j=0; j<kernel_width;j++) {
                        wtsLine[i] += ((kernel_height-1)*kernel_width*64); // Move to next ic/8 position
                        line[i] += (iw*8); // Increment to next ic/8 position (reset at end of outermost loop)

                    } // for(int i=kernel_height_start; i<kernel_height_end; i++) { // 1 to 3
                } // for(int ic=0; ic<(input_channels/8); ic++) {
                for(int x=0; x<8; x++) {
                    aie::vector<uint8,32> o1 = acc_tmp[x].to_vector<uint8>(scale);
                    aie::store_v(output, o1); output += 32;
                    acc_tmp[x] = aie::zeros<acc32,32>();   
                }
                // For next 8 activations, reset line buffer and weights
                for(int i=kernel_height_start; i<kernel_height_end; i++) {
                    line[i]    -= (input_channels/8)*(iw*8); // length of act to shift back
                }
            } // for(int iw_32c=0; iw_32c<iw_32; iw_32c++) {
            output += (iw_32_rem*32 + 32); // Shift past remainder output and left section of next oc/8
        } //     for(int oc=0; oc<(output_channels/8); oc++) {

        // Reset weights and line buffers for last section of middle (or right side it there is no last section)
        for(int i=kernel_height_start; i<kernel_height_end; i++) {
            wtsLine[i] -= (output_channels/8)*(input_channels/8)*kernel_width*kernel_height*64; // kernel_width*kernel_height*8*8
            // TODO line already shifted back to next data
            line[i]    += iw_32*256; // 8*4*8, shift to beginnign of secondary loop    }
        }
        output -= (output_channels/8)*(iw*8) - (iw_32*32); // 32 = 4*8

        } // if(iw_32 > 0)

        if (iw_32_rem > 0) {

        // Secondary loop for input_width remainder (iw_32_rem < 8)
        for(int oc=0; oc<(output_channels/8); oc++) {
            for(int ic=0; ic<(input_channels/8); ic++) {            
                for(int i=kernel_height_start; i<kernel_height_end; i++) { // 1 to 3
                    for(int j=0; j<kernel_width;j++) {
                        // New weight every kernel_width
                        aie::vector<int8, 64> wtsVec = aie::load_v<64>(wtsLine[i]); wtsLine[i] += 64;
                        // auto prev = prev_a[i].extract<32>(1);                  // prev = x0..x3(ci0..ci7)
                        auto prev = aie::load_v<32>((line[i]-32));
                        auto curr = aie::load_v<32>((line[i])); line[i]+=32;
                        auto next = aie::load_v<32>((line[i])); line[i]+=32;

                        for(int x=0; x<iw_32_rem; x++) 
                            // chess_unroll_loop()
                        {
                            auto tmp1   = aie::concat(curr,next);
                            auto tprev  = aie::concat(zero32, prev);
                            auto tmp2   = aie::shuffle_up_fill(tmp1, tprev, 8); // curr      = x3..x6(ci0..ci7)
                            auto tmp3   = aie::shuffle_down(tmp2, j*8);         // curr      = x4..x7(ci0..ci7) to x5..x8(ci0..ci7)ss

                            prev = curr;
                            curr = next;
                            next = aie::load_v<32>(line[i]); line[i]+=32; // next_prev = x4..x7(ci0..ci7)

                            acc_tmp[x].mac(tmp3.extract<32>(0), wtsVec);
                        }
                        line[i] -= (iw_32_rem+2)*32; // Reset line buffer ptr to beginning of line (after first 4)
                    }  //  for(int j=0; j<kernel_width;j++)
                    wtsLine[i] += ((kernel_height-1)*kernel_width*64); // Move to next ic/8 position
                    line[i] += (iw*8); // Increment to next ic/8 position (reset at end of outermost loop)
                } // for(int i=kernel_height_start; i<kernel_height_end; i++)
                // For next 8 input channels, line buffer and weights are automatically incremented to the right offset
            } // for(int ic=0; ic<(input_channels/8); ic++)
            // Write output from accumulator 
            for(int x=0; x<iw_32_rem; x++) {
                aie::vector<uint8,32> o1 = acc_tmp[x].to_vector<uint8>(scale);
                aie::store_v(output, o1); output += 32;
                acc_tmp[x] = aie::zeros<acc32,32>();   // Reset accumulators
            }
            // Output ptr should be in the right place (next oc/8)
            for(int i=kernel_height_start; i<kernel_height_end; i++) {
                line[i]    -= (input_channels/8)*(iw*8); // shift back to beginning of this section
            }
            output += (iw*8) - (iw_32_rem*32); // 32 = 4*8, shift to next oc/8
        } // for(int oc=0; oc<(output_channels/8); oc++)
        // Reset weights and line buffers for right side
        for(int i=kernel_height_start; i<kernel_height_end; i++) {
            wtsLine[i] -= (output_channels/8)*(input_channels/8)*kernel_width*kernel_height*64; // kernel_width*kernel_height*8*8
            line[i]    += iw_32_rem*32; // shift to beginnign of right data, iw_32_rem*4*8
        }
        // TODO
        output -= (output_channels/8)*(iw*8)-(iw_32_rem*32); // shift back so we're aligned with beginning of first oc/8 right

        } // if (iw_32_rem > 0) {

        // Right patterns
        {
            MMUL4x8x8 acc1 = aie::zeros<acc32,32>();
            for(int oc=0; oc<(output_channels/8); oc++) {
                for(int ic=0; ic<(input_channels/8); ic++) {
                    for(int i=kernel_height_start; i<kernel_height_end; i++)
                        chess_prepare_for_pipelining chess_loop_range(2,)    
                        // chess_unroll_loop()
                    {
                        // Load next set of data for input A (matrix row), need stride info or line1/2/3 pointer
                        // TODO, did not store previous so need to load it again
                        // in_a   = aie::load_v<64>(line[i]-32);
                        tmp_a1 = aie::load_v<32>(line[i]-32);
                        tmp_a2 = aie::load_v<32>(line[i]);
                        in_a   = aie::concat(tmp_a1, tmp_a2);
    #ifdef BORDER_REPLICATE 
                        tmp_a2    = aie::shuffle_down(tmp_a2, 24);
                        tmp_a.insert<32>(0,tmp_a2);
    #else
                        tmp_a     = aie::zeros<int8, 64>();
    #endif
                        in_a      = aie::shuffle_down_fill(in_a, tmp_a, 24); // shift by 32-8 (fill 32 then shift up by 8)
                    
                        for(int j=0; j<kernel_width; j++)
                            // chess_unroll_loop()
                        {
                            in_b  = aie::load_v<64>(wtsLine[i]); wtsLine[i] +=64; // wts ic0..7(oc0..7)
                            acc1.mac(in_a.extract<32>(0), in_b);
                            // Shift input A by 1 row (1x8) which is by 1 (the 8 is the ic=8)
                            in_a = aie::shuffle_down(in_a, 8);
                        }
                        wtsLine[i] += ((kernel_height-1)*kernel_width*64); // Move to next ic/8 position
                        // No need to load next set of weights because next row of weights immediately follows
                        line[i] += (iw*8); // Increment to next ic/8 position (reset at end of outermost loop)
                    } // for(int i=kernel_height_start; i<kernel_height_end; i++)
                } // for(int ic=0; ic<(input_channels/8); ic++) {

                // Write output 4 outputs, 8 channels
                aie::vector<uint8,32> o1 = acc1.to_vector<uint8>(scale);
                aie::store_v(output, o1); 
                acc1 = aie::zeros<acc32,32>();
                output += iw*8; // Shift to next oc/8

                for(int i=kernel_height_start; i<kernel_height_end; i++) {
                    line[i]    -= (input_channels/8)*(iw*8); // shift back to beginning of this section
                }
            } // for(int oc=0; oc<(output_channels/8); oc++) {
        }
        event1();
    }     
}



