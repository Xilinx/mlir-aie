//===- conv2dk1.cc -------------------------------------------------*- C++
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

// Factory dimensions are constants; raw-source callers keep runtime bounds.
#ifndef CONV_INPUT_WIDTH
#define CONV_INPUT_WIDTH runtime_input_width
#endif
#ifndef CONV_INPUT_CHANNELS
#define CONV_INPUT_CHANNELS runtime_input_channels
#endif
#ifndef CONV_OUTPUT_CHANNELS
#define CONV_OUTPUT_CHANNELS runtime_output_channels
#endif

#define REL_WRITE 0
#define REL_READ 1

#ifdef SCALAR

const int32_t UMAX = 255;

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t runtime_input_width,
                        const int32_t runtime_input_channels,
                        const int32_t runtime_output_channels,
                        const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
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

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t runtime_input_width,
                         const int32_t runtime_input_channels,
                         const int32_t runtime_output_channels,
                         const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
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

#endif // UINT8_ACT

#else // Vector

#if AIE_TUNED_AIE2
//*****************************************************************************
// conv2d 1x1 - vector
// act: int8 or uint8, wts: int8, out: uint8
//
// input_width must be a multiple of 32: each block of 32 pixels keeps 8
// accumulators of 4 pixels.
//*****************************************************************************
template <typename ActT>
static void
conv2dk1_vector(ActT *input, int8_t *kernels, uint8_t *__restrict output,
                const int32_t runtime_input_width,
                const int32_t runtime_input_channels,
                const int32_t runtime_output_channels, const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, ActT, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  constexpr int NUM_ACC = 8;
  const int iw = input_width;
  const int iw_32 = (input_width / 4) / 8;

  uint8_t *restrict out_ptr = output;

  for (int oc = 0; oc < (output_channels / 8); oc++) {
    for (int x = 0; x < iw_32; x++) {
      MMUL4x8x8 acc[NUM_ACC];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < NUM_ACC; i++)
        acc[i] = aie::zeros<acc32, 32>();
      // Two pointers, one per half block, so the loads can dual-issue.
      const ActT *restrict in0 = input + x * 256;
      const ActT *restrict in1 = in0 + 128;
      const int8_t *restrict w = kernels;
      // Rolled: LLVM's default unroll by two pipelines worse.
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_NO_UNROLL
      for (int ic = 0; ic < (input_channels / 8); ic++) {
        aie::vector<int8, 64> b = aie::load_v<64>(w);
        w += 64;
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC / 2; x8++) {
          acc[x8].mac(aie::load_v<32>(in0 + x8 * 32), b);
          acc[x8 + 4].mac(aie::load_v<32>(in1 + x8 * 32), b);
        }
        in0 += iw * 8;
        in1 += iw * 8;
      }
      AIE_LOOP_UNROLL_FULL
      for (int x8 = 0; x8 < NUM_ACC; x8++) {
        aie::store_v(out_ptr, acc[x8].template to_vector<uint8>(scale));
        out_ptr += 32;
      }
    }
    kernels += (input_channels / 8) * 64; // next oc/8 weights
  }

  event1();
}
#elif AIE_TUNED_AIE2P
//*****************************************************************************
// conv2d 1x1 - vector
// act: int8 or uint8, wts: int8, out: uint8
//
// input_width must be a multiple of 32: each block of 32 pixels keeps 4
// accumulators of 8 pixels, one native 8x8x8 mac each.
//*****************************************************************************
template <typename ActT>
static void
conv2dk1_vector(ActT *__restrict input, int8_t *__restrict kernels,
                uint8_t *__restrict output, const int32_t runtime_input_width,
                const int32_t runtime_input_channels,
                const int32_t runtime_output_channels, const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  event0();

  using MMUL8x8x8 = aie::mmul<8, 8, 8, ActT, int8>;
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  constexpr int NUM_ACC = 4;
  const int iw = input_width;
  const int iw_32 = input_width / 32;

  uint8_t *__restrict out_ptr = output;

  for (int oc = 0; oc < (output_channels / 8); oc++) {
    for (int x = 0; x < iw_32; x++) {
      MMUL8x8x8 acc[NUM_ACC];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < NUM_ACC; i++)
        acc[i] = aie::zeros<acc32, 64>();
      const ActT *__restrict in = input + x * 256;
      const int8_t *__restrict w = kernels;
      AIE_LOOP_UNROLL(8)
      for (int ic = 0; ic < (input_channels / 8); ic++) {
        aie::vector<int8, 64> b = aie::load_v<64>(w);
        w += 64;
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          acc[x8].mac(aie::load_v<64>(in + x8 * 64), b);
        in += iw * 8;
      }
      AIE_LOOP_UNROLL_FULL
      for (int x8 = 0; x8 < NUM_ACC; x8++) {
        aie::store_v(out_ptr, acc[x8].template to_vector<uint8>(scale));
        out_ptr += 64;
      }
    }
    kernels += (input_channels / 8) * 64; // next oc/8 weights
  }

  event1();
}
#else
#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - vector
// act: int8, wts: int8, out: uint8
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
void conv2dk1_i8_vector(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t runtime_input_width,
                        const int32_t runtime_input_channels,
                        const int32_t runtime_output_channels,
                        const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  event0();

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

  // TODO: keeping this variable produces incorrect results and a worse
  // schedule.
  const int iw = input_width;
  const int iw_32 = (input_width / 4) / 8;

  // const int iw_32_rem = (input_width / 4) % 8;
  // const int iw_32_rem = (32 / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int iw_32c = 0; iw_32c < iw_32; iw_32c++) {
        AIE_PREPARE_FOR_PIPELINING
        for (int ic = 0; ic < (input_channels / 8); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x = 0; x < 8; x++) {
            aie::vector<int8, 32> in_a = aie::load_v<32>(input);
            input += 32; // act oc0..3(ic0..7)
            acc_tmp[x].mac(in_a, in_b);
          }
          input += (iw * 8) - 256; // Move to next ic/8 position
        }
        // input ptr just moves to next section
        for (int xx = 0; xx < 8; xx++) {
          aie::vector<uint8, 32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
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
      AIE_PREPARE_FOR_PIPELINING
      for (int ic = 0; ic < (ics / 8); ic++) {
        aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
        kernels += 64; // wts ic0..7(oc0..7)

        for (int x = 0; x < iw_32_rem; x++) {
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

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 - vector
// act: uint8, wts: int8, out: uint8
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
void conv2dk1_ui8_vector(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t runtime_input_width,
                         const int32_t runtime_input_channels,
                         const int32_t runtime_output_channels,
                         const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
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

  // TODO: keeping this variable produces incorrect results and a worse
  // schedule.
  const int iw = input_width;
  const int iw_32 = (input_width / 4) / 8;

  // const int iw_32_rem = (input_width / 4) % 8;
  // const int iw_32_rem = (32 / 4) % 8;
  assert((input_width / 4) % 8 == 0);
  const int iw_32_rem = 0; // TODO - See restriction

  assert((input_channels / 8) > 2); // Assume IC >= 16

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int iw_32c = 0; iw_32c < iw_32; iw_32c++) {
        AIE_PREPARE_FOR_PIPELINING
        for (int ic = 0; ic < (input_channels / 8); ic++) {
          aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
          kernels += 64; // wts ic0..7(oc0..7)

          for (int x = 0; x < 8; x++) {
            aie::vector<uint8, 32> in_a = aie::load_v<32>(input);
            input += 32; // act oc0..3(ic0..7)
            acc_tmp[x].mac(in_a, in_b);
          }
          input += (iw * 8) - 256; // Move to next ic/8 position
        }
        // input ptr just moves to next section
        for (int xx = 0; xx < 8; xx++) {
          aie::vector<uint8, 32> o1 = acc_tmp[xx].to_vector<uint8>(scaleT);
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
      AIE_PREPARE_FOR_PIPELINING
      for (int ic = 0; ic < (ics / 8); ic++) {
        aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
        kernels += 64; // wts ic0..7(oc0..7)

        for (int x = 0; x < iw_32_rem; x++) {
          aie::vector<uint8, 32> in_a = aie::load_v<32>(input);
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

  event1();
}

#endif // UINT8_ACT
#endif

#endif // Vector

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

#else // UINT8_ACT

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                      output_channels, scale);
}

#endif // UINT8_ACT

#else // Vector

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t runtime_input_width,
                 const int32_t runtime_input_channels,
                 const int32_t runtime_output_channels, const int scale) {
  conv2dk1_vector<int8_t>(input, kernels, output, runtime_input_width,
                          runtime_input_channels, runtime_output_channels,
                          scale);
}

#else // UINT8_ACT

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t runtime_input_width,
                  const int32_t runtime_input_channels,
                  const int32_t runtime_output_channels, const int scale) {
  conv2dk1_vector<uint8_t>(input, kernels, output, runtime_input_width,
                           runtime_input_channels, runtime_output_channels,
                           scale);
}

#endif // UINT8_ACT
#else
#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_vector(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}

#else // UINT8_ACT

void conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                  const int32_t input_width, const int32_t input_channels,
                  const int32_t output_channels, const int scale) {
  conv2dk1_ui8_vector(input, kernels, output, input_width, input_channels,
                      output_channels, scale);
}

#endif // UINT8_ACT
#endif

#endif // Vector

} // extern "C"
