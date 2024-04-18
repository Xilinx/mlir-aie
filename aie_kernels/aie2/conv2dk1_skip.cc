//===- conv2dk1_skip.cc -------------------------------------------------*- C++
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
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#ifdef SCALAR

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - scalar
// act: uint8, wts: int8, skip: int8, out: uint8
//*****************************************************************************
void conv2dk1_skip_i8_scalar(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                             uint8_t *output, int8_t *skip,
                             const int32_t input_width,
                             const int32_t input_channels,
                             const int32_t output_channels, const int scale,
                             const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;
        for (ic = 0; ic < input_channels / 16; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // for (ic2 = input_channels/16; ic2 < input_channels/8; ic2++) {
        for (ic2 = 0; ic2 < input_channels / 16; ic2++) {
          for (ic8b = 0; ic8b < 8; ic8b++) {
            // int val2 = input1[ic2 * input_width + x];
            int val2 = input1[(ic2 * input_width * 8) + (x * 8) +
                              ic8b]; // TODO ic2 should be shifted?
            // int k2 = kernels[oc * input_channels + ic2];
            int k2 = kernels[(oc * (input_channels / 8) * 64) +
                             ((ic2 + (input_channels / 16)) * 64) + (ic8b * 8) +
                             oc8];
            sum += val2 * k2;
          }
        }
        // scale for convolution
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        // sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // //clip

        // scale for residual
        // skip_temp=skip[oc * input_width + x];
        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs + skip_temp;
        // skip_sum= sum_srs;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > UMAX) ? UMAX
                                 : (skip_sum_srs_final < 0)
                                     ? 0
                                     : skip_sum_srs_final; // clip

        // output[oc * input_width + x] = skip_sum_srs_final_out;
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

        // output[oc * input_width + x] = sum;
        // output[oc * input_width + x] = sum+skip[oc * input_width + x];
      }
    }
  }

  // for (oc = 0; oc < output_channels; ++oc) {
  //         for (x = 0; x < input_width; ++x) {
  //             output[oc * input_width + x]=skip[oc * input_width + x];}
  // }

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - scalar
// act: uint8, wts: int8, skip: uint8, out: uint8
//*****************************************************************************
void conv2dk1_skip_ui8_scalar(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                              uint8_t *output, uint8_t *skip,
                              const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels, const int scale,
                              const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        uint8_t skip_temp = 0;
        for (ic = 0; ic < input_channels / 16; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            uint8_t val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        for (ic2 = 0; ic2 < input_channels / 16; ic2++) {
          for (ic8b = 0; ic8b < 8; ic8b++) {
            // int val2 = input1[ic2 * input_width + x];
            uint8_t val2 = input1[(ic2 * input_width * 8) + (x * 8) +
                                  ic8b]; // TODO ic2 should be shifted?
            // int k2 = kernels[oc * input_channels + ic2];
            int k2 = kernels[(oc * (input_channels / 8) * 64) +
                             ((ic2 + (input_channels / 16)) * 64) + (ic8b * 8) +
                             oc8];
            sum += val2 * k2;
          }
        }
        // scale for convolution
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip

        // scale for residual
        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs + skip_temp;

        // skip_sum= sum_srs;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > UMAX) ? UMAX
                                 : (skip_sum_srs_final < 0)
                                     ? 0
                                     : skip_sum_srs_final; // clip

        // output[oc * input_width + x] = skip_sum_srs_final_out;
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

        // output[oc * input_width + x] = sum;
        // output[oc * input_width + x] = sum+skip[oc * input_width + x];
      }
    }
  }

  // for (oc = 0; oc < output_channels; ++oc) {
  //         for (x = 0; x < input_width; ++x) {
  //             output[oc * input_width + x]=skip[oc * input_width + x];}
  // }

  event1();
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - vector
// act: uint8, wts: int8, skip: int8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
void conv2dk1_skip_i8_vector(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                             uint8_t *output, int8_t *skip,
                             const int32_t input_width,
                             const int32_t input_channels,
                             const int32_t output_channels, const int scale,
                             const int skip_scale) {
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t *restrict out_ptr = output;
  int8_t *i_out_ptr = (int8_t *)output;
  int8_t *restrict skip_ptr = skip;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  constexpr int NUM_ACC = 8;

  const int iw_32 = (input_width / 4) / 8;
  const int iw = input_width;
  // const int iw_32_rem = (input_width / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  int input_offset1 = 0;
  int input_offset2 = 0;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int x = 0; x < iw_32; x++) {
        MMUL4x8x8 acc_tmp[NUM_ACC];
        for (int i = 0; i < NUM_ACC; i++) {
          acc_tmp[i] = aie::zeros<acc32, 32>();
        }
        for (int ic = 0; ic < (input_channels / 16); ic++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
            kernels += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++) {
              aie::vector<uint8, 32> in_a =
                  aie::load_v<32>(input0 + input_offset1);
              input_offset1 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset1 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
        for (int ic = 0; ic < (input_channels / 16); ic++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
            kernels += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++) {
              aie::vector<uint8, 32> in_a =
                  aie::load_v<32>(input1 + input_offset2);
              input_offset2 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset2 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
        // input ptr just moves to next section
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::vector<int8, 32> skip1 = aie::load_v<32>(skip_ptr);
          skip_ptr += 32;

          aie::accum<acc32, 32> accj;
          accj.from_vector(skip1, 0);
          accj = aie::add(accj, acc_tmp[x8].to_vector<int8>(scaleT));
          // accj = aie::mac(accj, acc_tmp[x8].to_vector<int8>(scaleT),
          // (uint8_t)1);
          aie::vector<uint8, 32> o1 = accj.to_vector<uint8>(skip_scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
          // acc_tmp[x8] = aie::zeros<acc32,32>();
        }
        input_offset1 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset2 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
      }                                // for(int x=0; x<iw_32; x++) {
      // input_offset -= (iw_32) * 256; // 8*32, reset beginning of input ptr
      input_offset1 = 0;                    // reset beginning of input ptr
      input_offset2 = 0;                    // reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }                  // for(int oc=0; oc<(output_channels/8); oc++) {

    out_ptr -= output_channels *
               iw; // output_channels/8*iw_32*8*32 = 256/8*(iw/4/8)*8*32

    // for(int oc=0; oc<(output_channels/8); oc++) {
    //     for(int x=0; x<iw_32; x++) {
    //         for(int x8=0; x8<NUM_ACC; x8++) {
    //             // aie::vector<uint8,32> skip1 = aie::load_v<32>(skip_ptr);
    //             skip_ptr += 32; aie::vector<int8,32> skip1 =
    //             aie::load_v<32>(skip_ptr); skip_ptr += 32;
    //             // aie::vector<uint8,32> tmp   = aie::load_v<32>(out_ptr);
    //             aie::vector<int8,32> tmp   = aie::load_v<32>(i_out_ptr);
    //             i_out_ptr += 32; aie::accum<acc32,32> accj;
    //             accj.from_vector(skip1,0);
    //             accj = aie::mac(accj, tmp, (uint8_t)1);
    //             aie::vector<uint8,32> o3 =
    //             accj.to_vector<uint8>(skip_scaleT); aie::store_v(out_ptr,
    //             o3); out_ptr += 32;
    //         }
    //     }
    //     out_ptr += (iw_32_rem*32);
    //     skip_ptr += (iw_32_rem*32);
    // }

    out_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);
    skip_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);

  } // if(iw_32 > 0) {

  // **TODO** Move out_ptr and skip_ptr back to first oc/8 rem location

  // if(iw_32_rem > 0) {

  // const int ocs = output_channels;
  // const int ics = input_channels;

  // input_offset1 = 0; // TODO need to offset this to ic_32_rem position
  // input_offset2 = 0; // TODO need to offset this to ic_32_rem position

  // for(int oc=0; oc<(ocs/8); oc++) {
  //     for(int ic=0; ic<(ics/16); ic++) {
  //         // For ic = oc = 8, we can load all the weights in 1x 512b vec reg
  //         (2x 256b loads)
  //         // For ic > 8, we would load the next 64 weights that are
  //         ic8..15(oc0..7)
  //         // For oc > 8, we would load the next 64 weights after all the ic
  //         weights {OC}{IC}{IC8}{OC8} aie::vector<int8, 64> in_b =
  //         aie::load_v<64>(kernels); kernels+=64; // wts ic0..7(oc0..7)

  //         for(int x=0; x<iw_32_rem; x++)
  //             // chess_prepare_for_pipelining //chess_loop_range(7, )
  //             // e.g. 28/4 = 7
  //             // 13 cycles delay for vload.
  //             // 7 gives us 3 cycle inner loop.
  //             // 13 gave 1 cycle inner loop before partial load, not it only
  //             gets 2 cycles (not sure why?)
  //         {
  //             aie::vector<uint8, 32> in_a      =
  //             aie::load_v<32>(input0+input_offset1); input_offset1 += 32; //
  //             act oc0..3(ic0..7) acc_tmp[x].mac(in_a, in_b);
  //         }
  //         input_offset1 += (iw*8)-(iw_32_rem*32); // Move to next ic/8
  //         position, TODO -(iw_32_rem*8)??
  //     }
  //     for(int ic=0; ic<(ics/16); ic++) {
  //         // For ic = oc = 8, we can load all the weights in 1x 512b vec reg
  //         (2x 256b loads)
  //         // For ic > 8, we would load the next 64 weights that are
  //         ic8..15(oc0..7)
  //         // For oc > 8, we would load the next 64 weights after all the ic
  //         weights {OC}{IC}{IC8}{OC8} aie::vector<int8, 64> in_b =
  //         aie::load_v<64>(kernels); kernels+=64; // wts ic0..7(oc0..7)

  //         for(int x=0; x<iw_32_rem; x++)
  //             // chess_prepare_for_pipelining //chess_loop_range(7, )
  //             // e.g. 28/4 = 7
  //             // 13 cycles delay for vload.
  //             // 7 gives us 3 cycle inner loop.
  //             // 13 gave 1 cycle inner loop before partial load, not it only
  //             gets 2 cycles (not sure why?)
  //         {
  //             aie::vector<uint8, 32> in_a      =
  //             aie::load_v<32>(input1+input_offset2); input_offset2 += 32; //
  //             act oc0..3(ic0..7) acc_tmp[x].mac(in_a, in_b);
  //         }
  //         input_offset2 += (iw*8)-(iw_32_rem*32); // Move to next ic/8
  //         position
  //     }
  //     // input ptr just moves to next section
  //     for(int xx=0; xx<iw_32_rem; xx++) {
  //         // aie::vector<uint8,32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
  //         aie::vector<int8,32> o1 = acc_tmp[xx].to_vector<int8>(scaleT);
  //         // aie::store_v(out_ptr, o1); out_ptr += 32;
  //         aie::store_v(i_out_ptr, o1); i_out_ptr += 32;
  //         acc_tmp[xx] = aie::zeros<acc32,32>();
  //     }
  //     // input   -= ((ics-1)/8)*(iw*8)+(iw_32_rem*32); // reset to beginning
  //     of input ptr for remainder input_offset1   -= 448; // reset to
  //     beginning of input ptr for remainder input_offset2   -= 448; // reset
  //     to beginning of input ptr for remainder
  //     // kernel ptr already at next oc/8
  //     i_out_ptr += (iw*8)-(iw_32_rem*32);           // move to next oc/8
  //     (skip remainder section if present)
  // }

  // i_out_ptr -= output_channels*iw;

  // for(int oc=0; oc<(output_channels/8); oc++) {
  //     for(int x8=0; x8<NUM_ACC; x8++) {
  //         aie::vector<int8,32> skip1 = aie::load_v<32>(skip_ptr); skip_ptr +=
  //         32; aie::vector<int8,32> tmp   = aie::load_v<32>(i_out_ptr);
  //         aie::accum<acc32,32> accj;
  //         accj.from_vector(skip1,0);
  //         accj = aie::mac(accj, tmp, (uint8_t)1);
  //         aie::vector<uint8,32> o3 = accj.to_vector<uint8>(skip_scaleT);
  //         aie::store_v(out_ptr, o3); out_ptr += 32;
  //     }
  //     out_ptr += (iw*8)-(iw_32_rem*32);
  //     skip_ptr += (iw*8)-(iw_32_rem*32);
  // }

  // } // if(iw_32_rem > 0)

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 skip - vector
// act: uint8, wts: int8, skip: uint8, out: uint8
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
void conv2dk1_skip_ui8_vector(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                              uint8_t *output, uint8_t *skip,
                              const int32_t input_width,
                              const int32_t input_channels,
                              const int32_t output_channels, const int scale,
                              const int skip_scale) {
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t *restrict out_ptr = output;
  int8_t *i_out_ptr = (int8_t *)output;
  uint8_t *restrict skip_ptr = skip;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  constexpr int NUM_ACC = 8;

  const int iw_32 = (input_width / 4) / 8;
  const int iw = input_width;
  // const int iw_32_rem = (input_width / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  int input_offset1 = 0;
  int input_offset2 = 0;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int x = 0; x < iw_32; x++) {
        MMUL4x8x8 acc_tmp[NUM_ACC];
        for (int i = 0; i < NUM_ACC; i++) {
          acc_tmp[i] = aie::zeros<acc32, 32>();
        }
        for (int ic = 0; ic < (input_channels / 16); ic++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
            kernels += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++) {
              aie::vector<uint8, 32> in_a =
                  aie::load_v<32>(input0 + input_offset1);
              input_offset1 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset1 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
        for (int ic = 0; ic < (input_channels / 16); ic++)
          chess_prepare_for_pipelining chess_loop_range(2, ) {
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
            kernels += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++) {
              aie::vector<uint8, 32> in_a =
                  aie::load_v<32>(input1 + input_offset2);
              input_offset2 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset2 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
        // input ptr just moves to next section
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::vector<uint8, 32> skip1 = aie::load_v<32>(skip_ptr);
          skip_ptr += 32;

          aie::accum<acc32, 32> accj;
          accj.from_vector(skip1, 0);
          accj = aie::add(accj, acc_tmp[x8].to_vector<int8>(scaleT));
          // accj = aie::mac(accj, acc_tmp[x8].to_vector<int8>(scaleT),
          // (uint8_t)1);
          aie::vector<uint8, 32> o1 = accj.to_vector<uint8>(skip_scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
          // acc_tmp[x8] = aie::zeros<acc32,32>();
        }
        input_offset1 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset2 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
      }                                // for(int x=0; x<iw_32; x++) {
      // input_offset -= (iw_32) * 256; // 8*32, reset beginning of input ptr
      input_offset1 = 0;                    // reset beginning of input ptr
      input_offset2 = 0;                    // reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }                  // for(int oc=0; oc<(output_channels/8); oc++) {

    out_ptr -= output_channels *
               iw; // output_channels/8*iw_32*8*32 = 256/8*(iw/4/8)*8*32

    // for(int oc=0; oc<(output_channels/8); oc++) {
    //     for(int x=0; x<iw_32; x++) {
    //         for(int x8=0; x8<NUM_ACC; x8++) {
    //             // aie::vector<uint8,32> skip1 = aie::load_v<32>(skip_ptr);
    //             skip_ptr += 32; aie::vector<int8,32> skip1 =
    //             aie::load_v<32>(skip_ptr); skip_ptr += 32;
    //             // aie::vector<uint8,32> tmp   = aie::load_v<32>(out_ptr);
    //             aie::vector<int8,32> tmp   = aie::load_v<32>(i_out_ptr);
    //             i_out_ptr += 32; aie::accum<acc32,32> accj;
    //             accj.from_vector(skip1,0);
    //             accj = aie::mac(accj, tmp, (uint8_t)1);
    //             aie::vector<uint8,32> o3 =
    //             accj.to_vector<uint8>(skip_scaleT); aie::store_v(out_ptr,
    //             o3); out_ptr += 32;
    //         }
    //     }
    //     out_ptr += (iw_32_rem*32);
    //     skip_ptr += (iw_32_rem*32);
    // }

    out_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);
    skip_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);

  } // if(iw_32 > 0) {

  // **TODO** Move out_ptr and skip_ptr back to first oc/8 rem location

  // if(iw_32_rem > 0) {

  // const int ocs = output_channels;
  // const int ics = input_channels;

  // input_offset1 = 0; // TODO need to offset this to ic_32_rem position
  // input_offset2 = 0; // TODO need to offset this to ic_32_rem position

  // for(int oc=0; oc<(ocs/8); oc++) {
  //     for(int ic=0; ic<(ics/16); ic++) {
  //         // For ic = oc = 8, we can load all the weights in 1x 512b vec reg
  //         (2x 256b loads)
  //         // For ic > 8, we would load the next 64 weights that are
  //         ic8..15(oc0..7)
  //         // For oc > 8, we would load the next 64 weights after all the ic
  //         weights {OC}{IC}{IC8}{OC8} aie::vector<int8, 64> in_b =
  //         aie::load_v<64>(kernels); kernels+=64; // wts ic0..7(oc0..7)

  //         for(int x=0; x<iw_32_rem; x++)
  //             // chess_prepare_for_pipelining //chess_loop_range(7, )
  //             // e.g. 28/4 = 7
  //             // 13 cycles delay for vload.
  //             // 7 gives us 3 cycle inner loop.
  //             // 13 gave 1 cycle inner loop before partial load, not it only
  //             gets 2 cycles (not sure why?)
  //         {
  //             aie::vector<uint8, 32> in_a      =
  //             aie::load_v<32>(input0+input_offset1); input_offset1 += 32; //
  //             act oc0..3(ic0..7) acc_tmp[x].mac(in_a, in_b);
  //         }
  //         input_offset1 += (iw*8)-(iw_32_rem*32); // Move to next ic/8
  //         position, TODO -(iw_32_rem*8)??
  //     }
  //     for(int ic=0; ic<(ics/16); ic++) {
  //         // For ic = oc = 8, we can load all the weights in 1x 512b vec reg
  //         (2x 256b loads)
  //         // For ic > 8, we would load the next 64 weights that are
  //         ic8..15(oc0..7)
  //         // For oc > 8, we would load the next 64 weights after all the ic
  //         weights {OC}{IC}{IC8}{OC8} aie::vector<int8, 64> in_b =
  //         aie::load_v<64>(kernels); kernels+=64; // wts ic0..7(oc0..7)

  //         for(int x=0; x<iw_32_rem; x++)
  //             // chess_prepare_for_pipelining //chess_loop_range(7, )
  //             // e.g. 28/4 = 7
  //             // 13 cycles delay for vload.
  //             // 7 gives us 3 cycle inner loop.
  //             // 13 gave 1 cycle inner loop before partial load, not it only
  //             gets 2 cycles (not sure why?)
  //         {
  //             aie::vector<uint8, 32> in_a      =
  //             aie::load_v<32>(input1+input_offset2); input_offset2 += 32; //
  //             act oc0..3(ic0..7) acc_tmp[x].mac(in_a, in_b);
  //         }
  //         input_offset2 += (iw*8)-(iw_32_rem*32); // Move to next ic/8
  //         position
  //     }
  //     // input ptr just moves to next section
  //     for(int xx=0; xx<iw_32_rem; xx++) {
  //         // aie::vector<uint8,32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
  //         aie::vector<int8,32> o1 = acc_tmp[xx].to_vector<int8>(scaleT);
  //         // aie::store_v(out_ptr, o1); out_ptr += 32;
  //         aie::store_v(i_out_ptr, o1); i_out_ptr += 32;
  //         acc_tmp[xx] = aie::zeros<acc32,32>();
  //     }
  //     // input   -= ((ics-1)/8)*(iw*8)+(iw_32_rem*32); // reset to beginning
  //     of input ptr for remainder input_offset1   -= 448; // reset to
  //     beginning of input ptr for remainder input_offset2   -= 448; // reset
  //     to beginning of input ptr for remainder
  //     // kernel ptr already at next oc/8
  //     i_out_ptr += (iw*8)-(iw_32_rem*32);           // move to next oc/8
  //     (skip remainder section if present)
  // }

  // i_out_ptr -= output_channels*iw;

  // for(int oc=0; oc<(output_channels/8); oc++) {
  //     for(int x8=0; x8<NUM_ACC; x8++) {
  //         aie::vector<int8,32> skip1 = aie::load_v<32>(skip_ptr); skip_ptr +=
  //         32; aie::vector<int8,32> tmp   = aie::load_v<32>(i_out_ptr);
  //         aie::accum<acc32,32> accj;
  //         accj.from_vector(skip1,0);
  //         accj = aie::mac(accj, tmp, (uint8_t)1);
  //         aie::vector<uint8,32> o3 = accj.to_vector<uint8>(skip_scaleT);
  //         aie::store_v(out_ptr, o3); out_ptr += 32;
  //     }
  //     out_ptr += (iw*8)-(iw_32_rem*32);
  //     skip_ptr += (iw*8)-(iw_32_rem*32);
  // }

  // } // if(iw_32_rem > 0)

  event1();
}

#endif // UINT8_ACT

#endif // Vector

//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

void conv2dk1_skip_i8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                      uint8_t *output, int8_t *skip, const int32_t input_width,
                      const int32_t input_channels,
                      const int32_t output_channels, const int scale,
                      const int skip_scale) {
  conv2dk1_skip_i8_scalar(input0, input1, kernels, output, skip, input_width,
                          input_channels, output_channels, scale, skip_scale);
}

#else // UINT8_ACT

void conv2dk1_skip_ui8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                       uint8_t *output, uint8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_scalar(input0, input1, kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

void conv2dk1_skip_i8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                      uint8_t *output, int8_t *skip, const int32_t input_width,
                      const int32_t input_channels,
                      const int32_t output_channels, const int scale,
                      const int skip_scale) {
  conv2dk1_skip_i8_vector(input0, input1, kernels, output, skip, input_width,
                          input_channels, output_channels, scale, skip_scale);
}

#else // UINT8_ACT

void conv2dk1_skip_ui8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                       uint8_t *output, uint8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_vector(input0, input1, kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // UINT8_ACT

#endif // Vector

} // extern "C"