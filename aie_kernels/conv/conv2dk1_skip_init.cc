//===- conv2dk1_skip_init.cc -------------------------------------------------*-
// C++
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

#define REL_WRITE 0
#define REL_READ 1

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
#ifndef CONV_SKIP_INPUT_CHANNELS
#define CONV_SKIP_INPUT_CHANNELS runtime_input_channels_skip
#endif

#ifdef SCALAR

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;

//*****************************************************************************
// conv2d 1x1 skip init - scalar
// act: uint8, wts: int8, skip: SkipT (int8 or uint8), out: uint8
//*****************************************************************************
// Channel counts are consumed in whole steps (input_channels in 16s, output
// and skip channels in 8s); kernels.conv2dk1_skip_init rejects anything else.
template <typename SkipT>
static void conv2dk1_skip_init_scalar(
    uint8_t *input0, uint8_t *input1, int8_t *kernels, uint8_t *output,
    SkipT *skip, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t input_channels_skip,
    const int scale, const int skip_scale, const int scale_skip_conv) {
  event0();

  int x, ic, ic2, ic3, oc, oc8, ic8, ic8b, ic8c;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  const int skip_scaleT_conv = scale_skip_conv;
  const int wts_offset = output_channels * input_channels;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int sum_skip_conv = 0;
        int sum_skip_conv_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;
        for (ic = 0; ic < input_channels / 16; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        for (ic2 = 0; ic2 < input_channels / 16; ic2++) {
          for (ic8b = 0; ic8b < 8; ic8b++) {
            int val2 = input1[(ic2 * input_width * 8) + (x * 8) + ic8b];
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
        // skip convolution
        for (ic3 = 0; ic3 < input_channels_skip / 8; ic3++) {
          for (ic8c = 0; ic8c < 8; ic8c++) {
            int val3 = skip[(ic3 * input_width * 8) + (x * 8) + ic8c];
            int k3 = kernels[(oc * (input_channels_skip / 8) * 64) +
                             (ic3 * 64) + (ic8c * 8) + oc8 + wts_offset];
            sum_skip_conv += val3 * k3;
          }
        }
        sum_skip_conv_srs =
            (sum_skip_conv + (1 << (skip_scaleT_conv - 1))) >> skip_scaleT_conv;
        sum_skip_conv_srs = (sum_skip_conv_srs > MAX)    ? MAX
                            : (sum_skip_conv_srs < -MIN) ? -MIN
                                                         : sum_skip_conv_srs;
        // scale for residual
        skip_temp = sum_skip_conv_srs;
        skip_sum = sum_srs + skip_temp;
        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > UMAX) ? UMAX
                                 : (skip_sum_srs_final < 0)
                                     ? 0
                                     : skip_sum_srs_final; // clip

        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;
      }
    }
  }

  event1();
}

#else // Vector

//*****************************************************************************
// conv2d 1x1 skip init - vector
// act: uint8, wts: int8, skip: SkipT (int8 or uint8), out: uint8
//*****************************************************************************
#if AIE_TUNED_AIE2
// The ic loops are promised MinTrips trips. Two is what lets them run as
// pipelined hardware loops.
template <typename SkipT, int MinTrips>
static void conv2dk1_skip_init_blocks(
    uint8_t *input0, uint8_t *input1, int8_t *kernels,
    uint8_t *__restrict output, SkipT *__restrict skip,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t input_channels_skip,
    const int scale, const int skip_scale, const int scale_skip_conv) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  using MMULSkip = aie::mmul<4, 8, 8, SkipT, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  constexpr int NUM_ACC = 8;
  const int iw = input_width;
  const int iw_32 = (input_width / 4) / 8;
  // input0 and input1 each hold half the input channels; the weights of
  // input0's channels come first within each oc/8 group. The skip
  // projection's weights follow all of the main conv's.
  const int ic_half = input_channels / 16;
  const int ic_skip = input_channels_skip / 8;
  int8_t *kernels_skip = kernels + output_channels * input_channels;

  uint8_t *restrict out_ptr = output;

  for (int oc = 0; oc < (output_channels / 8); oc++) {
    for (int x = 0; x < iw_32; x++) {
      {
        MMUL4x8x8 acc[NUM_ACC];
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < NUM_ACC; i++)
          acc[i] = aie::zeros<acc32, 32>();
        const uint8_t *restrict in0 = input0 + x * 256;
        const uint8_t *restrict in1 = input1 + x * 256;
        const int8_t *restrict w0 = kernels;
        const int8_t *restrict w1 = kernels + ic_half * 64;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(MinTrips)
        for (int ic = 0; ic < ic_half; ic++) {
          aie::vector<int8, 64> b0 = aie::load_v<64>(w0);
          aie::vector<int8, 64> b1 = aie::load_v<64>(w1);
          w0 += 64;
          w1 += 64;
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<32>(in0 + x8 * 32), b0);
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<32>(in1 + x8 * 32), b1);
          in0 += iw * 8;
          in1 += iw * 8;
        }
        // The int8 conv result waits in the output buffer for the skip.
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          aie::store_v((int8_t *)out_ptr + x8 * 32,
                       acc[x8].template to_vector<int8>(scale));
      }
      {
        MMULSkip acc[NUM_ACC];
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < NUM_ACC; i++)
          acc[i] = aie::zeros<acc32, 32>();
        const SkipT *restrict in = skip + x * 256;
        const int8_t *restrict w = kernels_skip;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(MinTrips)
        for (int ic = 0; ic < ic_skip; ic++) {
          aie::vector<int8, 64> b = aie::load_v<64>(w);
          w += 64;
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<32>(in + x8 * 32), b);
          in += iw * 8;
        }
        // Each step runs over all eight accumulators before the next, so
        // the eight chains get registers of their own and overlap.
        aie::vector<int8, 32> vs[NUM_ACC];
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          vs[x8] = acc[x8].template to_vector<int8>(scale_skip_conv);
        aie::accum<acc32, 32> sum[NUM_ACC];
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          sum[x8].from_vector(aie::load_v<32>((int8_t *)out_ptr + x8 * 32), 0);
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          sum[x8] = aie::mac(sum[x8], vs[x8], (int8_t)1);
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::store_v(out_ptr, sum[x8].template to_vector<uint8>(skip_scale));
          out_ptr += 32;
        }
      }
    }
    kernels += (input_channels / 8) * 64; // next oc/8 weights
    kernels_skip += ic_skip * 64;         // next oc/8 skip weights
  }

  // Only whole 32-wide blocks are computed; kernels.conv2dk1_skip_init
  // rejects other widths.
}

template <typename SkipT>
static void conv2dk1_skip_init_vector(
    uint8_t *input0, uint8_t *input1, int8_t *kernels, uint8_t *output,
    SkipT *skip, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t input_channels_skip,
    const int scale, const int skip_scale, const int scale_skip_conv) {
  event0();
  if (input_channels >= 32 && input_channels_skip >= 16)
    conv2dk1_skip_init_blocks<SkipT, 2>(input0, input1, kernels, output, skip,
                                        input_width, input_channels,
                                        output_channels, input_channels_skip,
                                        scale, skip_scale, scale_skip_conv);
  else
    conv2dk1_skip_init_blocks<SkipT, 1>(input0, input1, kernels, output, skip,
                                        input_width, input_channels,
                                        output_channels, input_channels_skip,
                                        scale, skip_scale, scale_skip_conv);
  event1();
}
#elif AIE_TUNED_AIE2P
// Each block of 32 pixels keeps 4 accumulators of 8 pixels, one native
// 8x8x8 mac each. The activations are not __restrict: with it the compiler
// parks every oc-invariant input vector on the stack.
template <typename SkipT>
static void conv2dk1_skip_init_vector(
    uint8_t *input0, uint8_t *input1, int8_t *kernels, uint8_t *output,
    SkipT *skip, const int32_t runtime_input_width,
    const int32_t runtime_input_channels, const int32_t runtime_output_channels,
    const int32_t runtime_input_channels_skip, const int scale,
    const int skip_scale, const int scale_skip_conv) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  const int32_t input_channels_skip = CONV_SKIP_INPUT_CHANNELS;
  event0();

  using MMUL8x8x8 = aie::mmul<8, 8, 8, uint8, int8>;
  using MMULSkip = aie::mmul<8, 8, 8, SkipT, int8>;
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  constexpr int NUM_ACC = 4;
  const int iw = input_width;
  const int iw_32 = input_width / 32;
  // input0 and input1 each hold half the input channels; the weights of
  // input0's channels come first within each oc/8 group. The skip
  // projection's weights follow all of the main conv's.
  const int ic_half = input_channels / 16;
  const int ic_skip = input_channels_skip / 8;
  int8_t *kernels_skip = kernels + output_channels * input_channels;

  uint8_t *out_ptr = output;

  for (int oc = 0; oc < (output_channels / 8); oc++) {
    for (int x = 0; x < iw_32; x++) {
      aie::vector<int8, 64> conv[NUM_ACC];
      {
        MMUL8x8x8 acc[NUM_ACC];
        const uint8_t *in0 = input0 + x * 256;
        const uint8_t *in1 = input1 + x * 256;
        const int8_t *__restrict w0 = kernels;
        const int8_t *__restrict w1 = kernels + ic_half * 64;
        // The first step multiplies, so no accumulator starts from zero.
        {
          aie::vector<int8, 64> b0 = aie::load_v<64>(w0);
          aie::vector<int8, 64> b1 = aie::load_v<64>(w1);
          w0 += 64;
          w1 += 64;
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mul(aie::load_v<64>(in0 + x8 * 64), b0);
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<64>(in1 + x8 * 64), b1);
          in0 += iw * 8;
          in1 += iw * 8;
        }
        AIE_LOOP_UNROLL(8)
        for (int ic = 1; ic < ic_half; ic++) {
          aie::vector<int8, 64> b0 = aie::load_v<64>(w0);
          aie::vector<int8, 64> b1 = aie::load_v<64>(w1);
          w0 += 64;
          w1 += 64;
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<64>(in0 + x8 * 64), b0);
          AIE_LOOP_UNROLL_FULL
          for (int x8 = 0; x8 < NUM_ACC; x8++)
            acc[x8].mac(aie::load_v<64>(in1 + x8 * 64), b1);
          in0 += iw * 8;
          in1 += iw * 8;
        }
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          conv[x8] = acc[x8].template to_vector<int8>(scale);
      }
      MMULSkip acc[NUM_ACC];
      const SkipT *in = skip + x * 256;
      const int8_t *__restrict w = kernels_skip;
      {
        aie::vector<int8, 64> b = aie::load_v<64>(w);
        w += 64;
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          acc[x8].mul(aie::load_v<64>(in + x8 * 64), b);
        in += iw * 8;
      }
      AIE_LOOP_UNROLL(8)
      for (int ic = 1; ic < ic_skip; ic++) {
        aie::vector<int8, 64> b = aie::load_v<64>(w);
        w += 64;
        AIE_LOOP_UNROLL_FULL
        for (int x8 = 0; x8 < NUM_ACC; x8++)
          acc[x8].mac(aie::load_v<64>(in + x8 * 64), b);
        in += iw * 8;
      }
      AIE_LOOP_UNROLL_FULL
      for (int x8 = 0; x8 < NUM_ACC; x8++) {
        aie::accum<acc32, 64> accj;
        accj.from_vector(conv[x8], 0);
        accj = aie::mac(accj, acc[x8].template to_vector<int8>(scale_skip_conv),
                        (int8_t)1);
        aie::store_v(out_ptr, accj.template to_vector<uint8>(skip_scale));
        out_ptr += 64;
      }
    }
    kernels += (input_channels / 8) * 64; // next oc/8 weights
    kernels_skip += ic_skip * 64;         // next oc/8 skip weights
  }

  // Only whole 32-wide blocks are computed; kernels.conv2dk1_skip_init
  // rejects other widths.

  event1();
}
#else
template <typename SkipT>
static void conv2dk1_skip_init_vector(
    uint8_t *input0, uint8_t *input1, int8_t *kernels, uint8_t *output,
    SkipT *skip, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t input_channels_skip,
    const int scale, const int skip_scale, const int scale_skip_conv)

{
  event0();

  using MMUL4x8x8 = aie::mmul<4, 8, 8, uint8, int8>;
  using MMULSkip = aie::mmul<4, 8, 8, SkipT, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8

  uint8_t * /*restrict*/ out_ptr = output;
  int8_t *i_out_ptr = (int8_t *)output;
  SkipT *restrict skip_ptr = skip;

  const int wts_offset = output_channels * input_channels;
  int8_t *kernels_skip = kernels + wts_offset;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  const int scaleT_skip_conv = scale_skip_conv;

  constexpr int NUM_ACC = 8;

  const int iw_32 = (input_width / 4) / 8;
  const int iw = input_width;
  const int iw_32_rem = (input_width / 4) % 8;

  int input_offset1 = 0;
  int input_offset2 = 0;
  int input_offset3 = 0;

  if (iw_32 > 0) {

    for (int oc = 0; oc < (output_channels / 8); oc++) {
      for (int x = 0; x < iw_32; x++) {
        aie::vector<int8, 32> vec_conv[NUM_ACC];
        aie::vector<int8, 32> vec_skip[NUM_ACC];

        { // conv section
          MMUL4x8x8 acc_tmp[NUM_ACC];
          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            acc_tmp[x8] = aie::zeros<acc32, 32>();
          }

          for (int ic = 0; ic < (input_channels / 16); ic++) { // half ic/8
            // For ic = oc = 8, we can load all the weights in 1x 512b vec reg
            // (2x 256b loads) For ic > 8, we would load the next 64 weights
            // that are ic8..15(oc0..7) For oc > 8, we would load the next 64
            // weights after all the ic weights {OC}{IC}{IC8}{OC8}
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels);
            kernels += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++)
            // All four NUM_ACC loops carried the same commented-out Chess
            // pragma (chess_loop_range(7, )) and its tuning notes: a range of
            // 7 gave a 3-cycle inner loop, 13 gave 1 cycle before partial
            // loads and 2 after, against a vload costing about 13. Peano, the
            // toolchain this kernel builds with, ignores the pragma, so the
            // measurements are what is kept -- once, here.
            {
              aie::vector<uint8, 32> in_a =
                  aie::load_v<32>(input0 + input_offset1);
              input_offset1 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset1 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
          for (int ic = 0; ic < (input_channels / 16); ic++) { // half ic/8
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
          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            vec_conv[x8] = acc_tmp[x8].to_vector<int8>(scaleT);
          }
        } // conv section

        { // skip section
          MMULSkip acc_tmp[NUM_ACC];
          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            acc_tmp[x8] = aie::zeros<acc32, 32>();
          }

          for (int ic = 0; ic < (input_channels_skip / 8); ic++) {
            // For oc > 8, we would load the next 64 weights after all the ic
            // weights {OC}{IC}{IC8}{OC8}
            aie::vector<int8, 64> in_b = aie::load_v<64>(kernels_skip);
            kernels_skip += 64; // wts ic0..7(oc0..7)

            for (int x8 = 0; x8 < NUM_ACC; x8++) {
              aie::vector<SkipT, 32> in_a =
                  aie::load_v<32>(skip + input_offset3);
              input_offset3 += 32; // act oc0..3(ic0..7)
              acc_tmp[x8].mac(in_a, in_b);
            }
            input_offset3 +=
                (iw * 8) -
                256; // Move to next ic/8 position. 256 = 32 input * 8 ic
          }
          for (int x8 = 0; x8 < NUM_ACC; x8++) {
            vec_skip[x8] =
                acc_tmp[x8].template to_vector<int8>(scaleT_skip_conv);
          }
        } // skip section

        // input ptr just moves to next section
        for (int x8 = 0; x8 < NUM_ACC; x8++) {
          aie::accum<acc32, 32> accj;
          accj.from_vector(vec_conv[x8], 0);
          accj = aie::add(accj, vec_skip[x8]);
          aie::vector<uint8, 32> o1 = accj.to_vector<uint8>(skip_scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += 32;
        }
        input_offset1 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset2 -=
            ((input_channels / 16) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        input_offset3 -=
            ((input_channels_skip / 8) * iw * 8) -
            256; // reset to next input_width/32 block. 256 = 32 input * 8 ic
        kernels -=
            (input_channels / 8) * 64; // reset kernel back to beginning of ic/8
        kernels_skip -= (input_channels_skip / 8) *
                        64; // reset kernel back to beginning of ic/8
      } // for(int x=0; x<iw_32; x++) {
      input_offset1 = 0;                    // reset beginning of input ptr
      input_offset2 = 0;                    // reset beginning of input ptr
      input_offset3 = 0;                    // reset beginning of input ptr
      kernels += (input_channels / 8) * 64; // move to next oc/8 weights
      kernels_skip +=
          (input_channels_skip / 8) * 64; // move to next oc/8 weights
      out_ptr += (iw_32_rem *
                  32); // move to next oc/8 (skip remainder section if present)
    } // for(int oc=0; oc<(output_channels/8); oc++) {

    out_ptr -= output_channels *
               iw; // output_channels/8*iw_32*8*32 = 256/8*(iw/4/8)*8*32

    out_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);
    skip_ptr -= (output_channels - 1) * iw + (iw_32_rem * 32);

  } // if(iw_32 > 0) {

  // Only whole 32-wide blocks are computed. The iw_32_rem tail was never
  // implemented, so an input_width that is not a multiple of 32 leaves its
  // last (input_width % 32) columns unwritten rather than raising. The
  // factory rejects such a width; see kernels.conv2dk1_skip_init.

  event1();
}
#endif

#endif // Vector

//*****************************************************************************
// conv2d 1x1 skip init wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

void conv2dk1_skip_init_i8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                           uint8_t *output, int8_t *skip,
                           const int32_t input_width,
                           const int32_t input_channels,
                           const int32_t output_channels,
                           const int32_t input_channels_skip, const int scale,
                           const int skip_scale, const int scale_skip_conv) {
  conv2dk1_skip_init_scalar<int8_t>(
      input0, input1, kernels, output, skip, input_width, input_channels,
      output_channels, input_channels_skip, scale, skip_scale, scale_skip_conv);
}

#else // UINT8_ACT

void conv2dk1_skip_init_ui8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                            uint8_t *output, uint8_t *skip,
                            const int32_t input_width,
                            const int32_t input_channels,
                            const int32_t output_channels,
                            const int32_t input_channels_skip, const int scale,
                            const int skip_scale, const int scale_skip_conv) {
  conv2dk1_skip_init_scalar<uint8_t>(
      input0, input1, kernels, output, skip, input_width, input_channels,
      output_channels, input_channels_skip, scale, skip_scale, scale_skip_conv);
}

#endif // UINT8_ACT

#else // Vector

#ifdef INT8_ACT

void conv2dk1_skip_init_i8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                           uint8_t *output, int8_t *skip,
                           const int32_t input_width,
                           const int32_t input_channels,
                           const int32_t output_channels,
                           const int32_t input_channels_skip, const int scale,
                           const int skip_scale, const int scale_skip_conv) {
  conv2dk1_skip_init_vector<int8_t>(
      input0, input1, kernels, output, skip, input_width, input_channels,
      output_channels, input_channels_skip, scale, skip_scale, scale_skip_conv);
}

#else // UINT8_ACT

void conv2dk1_skip_init_ui8(uint8_t *input0, uint8_t *input1, int8_t *kernels,
                            uint8_t *output, uint8_t *skip,
                            const int32_t input_width,
                            const int32_t input_channels,
                            const int32_t output_channels,
                            const int32_t input_channels_skip, const int scale,
                            const int skip_scale, const int scale_skip_conv) {
  conv2dk1_skip_init_vector<uint8_t>(
      input0, input1, kernels, output, skip, input_width, input_channels,
      output_channels, input_channels_skip, scale, skip_scale, scale_skip_conv);
}

#endif // UINT8_ACT

#endif // Vector

} // extern "C"
