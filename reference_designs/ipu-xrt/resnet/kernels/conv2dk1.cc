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
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

#ifdef SCALAR

const int32_t UMAX = 255;

void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t input_width,
                         const int32_t input_channels,
                         const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            uint8_t val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int8_t k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                               (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#else // Vector

void conv2dk1_i8_vector(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t *restrict out_ptr = output;

  const int scaleT = scale;

  MMUL4x8x8 acc_tmp[8];
  for (int x = 0; x < 8; x++) {
    acc_tmp[x] = aie::zeros<acc32, 32>();
  }

  // const int iw_32     = (input_width/4)/8;
  const int iw_32 = (32 / 4) / 8;
  // const int iw_32_rem = (input_width/4)%8;
  // const int iw        = input_width;
  const int iw = 32;

  // const int iw_32_rem = (iw/4)%8;

  // const int iw_32     = (28/4)/8;
  const int iw_32_rem = (32 / 4) % 8; // TODO Change if input_width changes
  // const int iw_32_rem = (iw/4) - (iw_32*8);
  // const int iw        = 28;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int iw_32c = 0; iw_32c < iw_32; iw_32c++) {
        for (int ic = 0; ic < (input_channels / 8); ic++) {
          // For ic = oc = 8, we can load all the weights in 1x 512b vec reg (2x
          // 256b loads) For ic > 8, we would load the next 64 weights that are
          // ic8..15(oc0..7) For oc > 8, we would load the next 64 weights after
          // all the ic weights {OC}{IC}{IC8}{OC8}
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x = 0; x < 8; x++)
          // chess_prepare_for_pipelining //chess_loop_range(7, )
          // e.g. 28/4 = 7
          // 13 cycles delay for vload.
          // 7 gives us 3 cycle inner loop.
          // 13 gave 1 cycle inner loop before partial load, not it only gets 2
          // cycles (not sure why?)
          {
            aie::vector<int8, 32> in_a = aie::load_v<32>(input);
            input += 32; // act oc0..3(ic0..7)
            acc_tmp[x].mac(in_a, in_b);
          }
          input += (iw * 8) - 256; // Move to next ic/8 position
        }
        // input ptr just moves to next section
        for (int xx = 0; xx < 8; xx++) {
          aie::vector<uint8, 32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
          // aie::vector<uint8,32> o1 = aie::zeros<uint8,32>();
          // aie::vector<uint8,32> o1;
          // for(int i=0; i<32;i++) {
          //     o1[i] = i + (oc+1)*8;
          // }
          // aie::vector<uint8,32> o1 = aie::load_v<32>(input-256);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
          acc_tmp[xx] = aie::zeros<acc32, 32>();
        }
        input -= ((input_channels / 8) * iw * 8) -
                 256; // reset to next input_width/32 block
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
      }
      input -= (iw_32) * 256; // 8*32, reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }

  } // if(iw_32 > 0) {

  if (iw_32_rem > 0) {

    const int ocs = output_channels;
    const int ics = input_channels;

    for (int oc = 0; oc < (ocs / 8); oc++) {
      for (int ic = 0; ic < (ics / 8); ic++) {
        // For ic = oc = 8, we can load all the weights in 1x 512b vec reg (2x
        // 256b loads) For ic > 8, we would load the next 64 weights that are
        // ic8..15(oc0..7) For oc > 8, we would load the next 64 weights after
        // all the ic weights {OC}{IC}{IC8}{OC8}
        aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
        kernels += 64; // wts ic0..7(oc0..7)

        for (int x = 0; x < iw_32_rem; x++)
        // chess_prepare_for_pipelining //chess_loop_range(7, )
        // e.g. 28/4 = 7
        // 13 cycles delay for vload.
        // 7 gives us 3 cycle inner loop.
        // 13 gave 1 cycle inner loop before partial load, not it only gets 2
        // cycles (not sure why?)
        {
          aie::vector<int8, 32> in_a = aie::load_v<32>(input);
          input += 32; // act oc0..3(ic0..7)
          acc_tmp[x].mac(in_a, in_b);
        }
        input += (iw * 8) - (iw_32_rem * 32); // Move to next ic/8 position
      }
      // input ptr just moves to next section
      for (int xx = 0; xx < iw_32_rem; xx++) {
        aie::vector<uint8, 32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
        aie::store_v(out_ptr, o1);
        out_ptr += 32;
        acc_tmp[xx] = aie::zeros<acc32, 32>();
      }
      // input   -= ((ics-1)/8)*(iw*8)+(iw_32_rem*32); // reset to beginning of
      // input ptr for remainder
      input -= 448; // reset to beginning of input ptr for remainder
      // kernel ptr already at next oc/8
      out_ptr += (iw * 8) -
                 (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    }

  } // if(iw_32_rem > 0)
}

#endif

extern "C" {

#ifdef SCALAR

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                      output_channels, scale);
}

#else // Vector

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_vector(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  // TODO Missing conv2dk1_ui8 version
}

#endif

} // extern "C"
