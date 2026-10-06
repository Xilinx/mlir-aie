//===- conv2dk14.cc -------------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef CONV_INPUT_WIDTH
#define CONV_INPUT_WIDTH runtime_input_width
#endif
#ifndef CONV_INPUT_CHANNELS
#define CONV_INPUT_CHANNELS runtime_input_channels
#endif
#ifndef CONV_OUTPUT_CHANNELS
#define CONV_OUTPUT_CHANNELS runtime_output_channels
#endif
#ifndef CONV_KERNEL_WIDTH
#define CONV_KERNEL_WIDTH runtime_kernel_width
#endif

#define REL_WRITE 0
#define REL_READ 1

#ifdef SCALAR

const int32_t SMAX = 127;
const int32_t SMIN = 128;

#ifdef INT8_ACT

#else // UINT8_ACT

//*****************************************************************************
// conv2d 3x3 - scalar
// act: int8, wts: int8, out: uint8
//
// Input - t/8 p/2 t8 p2  --> 2 98 8 2
//     t/8 is tiles/ 8 but we're only processing 16 tiles or t/2 = 2
//     p/2 is pixels / 2 of which we have 14x14 or 196 pixesl. p/2 = 98
//     t8 is 8 tiles
//     p2 is 2 pixels and a pixel is rgba (32bit)
//
// Kernel weights - ch/8 p/2 p2 ch8 --> 2 98 2 8
//     ch8 is 8 channels
//     ch/8 is channels divided by 8 but we only process 16 channels so ch/8=2
//
// Output - ch/8 t/8 t8 c8 --> 2 2 8 8
//
//*****************************************************************************
void conv2dk14_i8_scalar(uint8_t *input, int8_t *kernels, int8_t *output,
                         const int32_t runtime_input_width,
                         const int32_t runtime_input_channels,
                         const int32_t runtime_output_channels,
                         const int32_t runtime_kernel_width, const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  const int32_t kernel_width = CONV_KERNEL_WIDTH;
  event0();

  int oc, oc8, nt, nt8, pix, p2;

  int in_indx = 0;
  int wts_indx = 0;
  int out_indx = 0;

  const int output_channels_div_8 = output_channels / 8;    // 2
  const int tiles_div_8 = input_width / kernel_width / 8;   // 2
  const int pixels_div_2 = kernel_width * kernel_width / 2; // 98

  for (oc = 0; oc < output_channels_div_8; oc++) { // 16 out of 1152
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (nt = 0; nt < tiles_div_8; nt++) { // 16 out of 64 tiles in row
        for (nt8 = 0; nt8 < 8; nt8++) {
          int sum = 0;
          for (pix = 0; pix < pixels_div_2; pix++) { // 196 // 2 = 98
            for (p2 = 0; p2 < 2; p2++) {
              in_indx = ((nt * (pixels_div_2) * 8 * 2) + (pix * 8 * 2) +
                         (nt8 * 2) + p2) *
                        4;
              wts_indx = ((oc * pixels_div_2 * 2 * 4 * 8) + (pix * 2 * 4 * 8) +
                          (p2 * 4 * 8) + oc8);
              sum += input[in_indx] * kernels[wts_indx] +
                     input[in_indx + 1] * kernels[wts_indx + 8] +
                     input[in_indx + 2] * kernels[wts_indx + 16] +
                     input[in_indx + 3] * kernels[wts_indx + 24];
            }
          }
          int sum_srs = (sum + (1 << (scale - 1))) >> scale;
          sum_srs = (sum_srs > SMAX)    ? SMAX
                    : (sum_srs < -SMIN) ? -SMIN
                                        : sum_srs;
          out_indx =
              (oc * (tiles_div_8) * 8 * 8) + (nt * 8 * 8) + (nt8 * 8) + oc8;
          output[out_indx] = sum_srs;
        }
      }
    }
  }

  event1();
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

#else // UINT8_ACT

//*****************************************************************************
// conv2d 3x3 - vector
// act: int8, wts: int8, out: uint8
//
// Input - t/8 p/2 t8 p2  --> 2 98 8 2
//     t/8 is tiles/ 8 but we're only processing 16 tiles or t/2 = 2
//     p/2 is pixels / 2 of which we have 14x14 or 196 pixesl. p/2 = 98
//     t8 is 8 tiles
//     p2 is 2 pixels and a pixel is rgba (32bit)
//
// Kernel weights - ch/8 p/2 p2 ch8 --> 2 98 2 8
//     ch8 is 8 channels
//     ch/8 is channels divided by 8 but we only process 16 channels so ch/8=2
//
// Output - ch/8 t/8 t8 c8 --> 2 2 8 8
//
//*****************************************************************************
#if AIE_TUNED_AIE2
void conv2dk14_i8_vector(uint8_t *input, int8_t *kernels, int8_t *output,
                         const int32_t runtime_input_width,
                         const int32_t runtime_input_channels,
                         const int32_t runtime_output_channels,
                         const int32_t runtime_kernel_width, const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  const int32_t kernel_width = CONV_KERNEL_WIDTH;
  event0();

  // Each 8 tiles x 8 channels block is two native 4x8x8 macs on the two
  // 256-bit halves of the activations, so no shuffle is needed.
  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  const int output_channels_div_8 = output_channels / 8;
  const int tiles_div_16 = input_width / kernel_width / 16;
  const int pixels_div_2 = kernel_width * kernel_width / 2;

  const int group = tiles_div_16 * 128;
  int8_t *__restrict out_ptr = output;

  // Two channel groups share each activation load. An odd last group is
  // computed twice and stored once.
  for (int k = 0; k < output_channels_div_8; k += 2) {
    const int second = k + 1 < output_channels_div_8 ? group : 0;
    const uint8_t *__restrict in_ptr = input;
    for (int j = 0; j < tiles_div_16; j++) {
      const uint8_t *__restrict a_ptr = in_ptr;
      const uint8_t *__restrict b_ptr = in_ptr + pixels_div_2 * 64;
      const int8_t *__restrict k_ptr = kernels + k * pixels_div_2 * 64;
      const int8_t *__restrict l_ptr =
          second ? k_ptr + pixels_div_2 * 64 : k_ptr;
      MMUL4x8x8 acc0 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc1 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc2 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc3 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc4 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc5 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc6 = aie::zeros<acc32, 32>();
      MMUL4x8x8 acc7 = aie::zeros<acc32, 32>();
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_NO_UNROLL
      for (int i = 0; i < pixels_div_2; i++) {
        aie::vector<int8, 64> w0 = aie::load_v<64>(k_ptr);
        aie::vector<int8, 64> w1 = aie::load_v<64>(l_ptr);
        k_ptr += 64;
        l_ptr += 64;
        aie::vector<uint8, 32> x0 = aie::load_v<32>(a_ptr);
        aie::vector<uint8, 32> x1 = aie::load_v<32>(a_ptr + 32);
        aie::vector<uint8, 32> x2 = aie::load_v<32>(b_ptr);
        aie::vector<uint8, 32> x3 = aie::load_v<32>(b_ptr + 32);
        a_ptr += 64;
        b_ptr += 64;
        acc0.mac(x0, w0);
        acc1.mac(x1, w0);
        acc2.mac(x2, w0);
        acc3.mac(x3, w0);
        acc4.mac(x0, w1);
        acc5.mac(x1, w1);
        acc6.mac(x2, w1);
        acc7.mac(x3, w1);
      }
      aie::store_v(out_ptr, acc0.to_vector<int8>(scale));
      aie::store_v(out_ptr + 32, acc1.to_vector<int8>(scale));
      aie::store_v(out_ptr + 64, acc2.to_vector<int8>(scale));
      aie::store_v(out_ptr + 96, acc3.to_vector<int8>(scale));
      if (second) {
        aie::store_v(out_ptr + second, acc4.to_vector<int8>(scale));
        aie::store_v(out_ptr + second + 32, acc5.to_vector<int8>(scale));
        aie::store_v(out_ptr + second + 64, acc6.to_vector<int8>(scale));
        aie::store_v(out_ptr + second + 96, acc7.to_vector<int8>(scale));
      }
      out_ptr += 128;
      in_ptr += 2 * pixels_div_2 * 64;
    }
    out_ptr += second;
  }

  event1();
}
#else
void conv2dk14_i8_vector(uint8_t *input, int8_t *kernels, int8_t *output,
                         const int32_t runtime_input_width,
                         const int32_t runtime_input_channels,
                         const int32_t runtime_output_channels,
                         const int32_t runtime_kernel_width, const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  const int32_t kernel_width = CONV_KERNEL_WIDTH;
  event0();

  // Compute
  using MMUL8x8x8 = aie::mmul<8, 8, 8, uint8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::symmetric_inf); // Needed to saturate properly to int8

  aie::vector<int8, 64> zero64 = aie::zeros<int8, 64>();

  MMUL8x8x8 acc1 = aie::zeros<acc32, 64>();
  MMUL8x8x8 acc2 = aie::zeros<acc32, 64>();
  aie::vector<int8, 64> maxv = aie::broadcast<int8, 64>(127);

  const int output_channels_div_8 = output_channels / 8;    // 2
  const int tiles_div_8 = input_width / kernel_width / 8;   // 2
  const int tiles_div_16 = input_width / kernel_width / 16; // 1
  const int pixels_div_2 = kernel_width * kernel_width / 2; // 98

  uint8_t *__restrict in_ptr_1 = input;
  uint8_t *__restrict in_ptr_2 = input + pixels_div_2 * 64;
  int8_t *__restrict k_ptr = kernels;
  int8_t *__restrict out_ptr = output;

  for (int k = 0; k < output_channels_div_8; k++) { // 2
    for (int j = 0; j < tiles_div_16; j++) {        // 1
      AIE_PREPARE_FOR_PIPELINING
      // AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < pixels_div_2; i++) {   // 98
        auto tmp_a1 = aie::load_v<64>(in_ptr_1); // 8 tiles x 2 pixels
        in_ptr_1 += 64;
        auto tmp_a2 = aie::load_v<64>(k_ptr); // 2 pixels x 8 channels
        acc1.mac(tmp_a1, tmp_a2); // 8 tiles x 8 channels (for 2 pixels)
        auto tmp_b1 = aie::load_v<64>(in_ptr_2); // 8 tiles x 2 pixels
        in_ptr_2 += 64;
        acc2.mac(tmp_b1, tmp_a2); // 8 tiles x 8 channels (for 2 pixels)
        k_ptr += 64;
      }
      aie::vector<int8, 64> o1 = acc1.to_vector<int8>(scale);
      aie::vector<int8, 64> o2 = acc2.to_vector<int8>(scale);
      aie::store_v(out_ptr, o1);
      out_ptr += 64;
      aie::store_v(out_ptr, o2);
      out_ptr += 64;
      acc1 = aie::zeros<acc32, 64>();
      acc2 = aie::zeros<acc32, 64>();
      k_ptr -= 64 * pixels_div_2;
      in_ptr_1 += pixels_div_2 * 64;
      in_ptr_2 += pixels_div_2 * 64;
    }
    k_ptr += 64 * pixels_div_2;
    in_ptr_1 -= 2 * tiles_div_16 * pixels_div_2 * 64;
    in_ptr_2 -= 2 * tiles_div_16 * pixels_div_2 * 64;
  }

  event1();
}
#endif

#endif // UINT8_ACT

#endif // Vector

extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

// void conv2dk14_i8(int8_t *input, int8_t *kernels, int8_t *output,
//                   const int32_t input_width, const int32_t input_channels,
//                   const int32_t output_channels, const int32_t kernel_width,
//                   const int scale) {
//   conv2dk14_i8_scalar(input, kernels, output, input_width, input_channels,
//                       output_channels, kernel_width, scale);
// }

#else // UINT8_ACT

void conv2dk14_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int scale) {
  conv2dk14_i8_scalar(input, kernels, output, input_width, input_channels,
                      output_channels, kernel_width, scale);
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

// void conv2dk14_i8(int8_t *input, int8_t *kernels, int8_t *output,
//                   const int32_t input_width, const int32_t input_channels,
//                   const int32_t output_channels, const int32_t kernel_width,
//                   const int scale) {
//   conv2dk14_i8_vector(input, kernels, output, input_width, input_channels,
//                       output_channels, kernel_width, scale);
// }

#else // UINT8_ACT

void conv2dk14_i8(uint8_t *input, int8_t *kernels, int8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int scale) {
  conv2dk14_i8_vector(input, kernels, output, input_width, input_channels,
                      output_channels, kernel_width, scale);
}

#endif // UINT8_ACT

#endif // Vector
}
