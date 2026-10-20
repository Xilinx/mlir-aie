//===- conv2dk1_i8.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include "../common/table_i8.h"
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

// Input width split across the MMUL; the product is 32 either way.
#if AIE_TUNED_AIE2
#define CONV2DK1_I8_NUM_ACC 8
#define CONV2DK1_I8_MMUL_M 4
#else
#define CONV2DK1_I8_NUM_ACC 4
#define CONV2DK1_I8_MMUL_M 8
#endif

// The factory names each variant's entry point.
#ifndef CONV2DK1_I8_SYMBOL
#define CONV2DK1_I8_SYMBOL conv2dk1_i8
#endif

#ifdef CONV_ROUND_EVEN
#define CONV2DK1_I8_ROUNDING aie::rounding_mode::conv_even
#else
#define CONV2DK1_I8_ROUNDING aie::rounding_mode::symmetric_inf
#endif

#define REL_WRITE 0
#define REL_READ 1

#ifdef SCALAR

const int32_t SMAX = 127;
const int32_t SMIN = 128;

#ifdef INT8_ACT
//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: int8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, int8_t *output,
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
        sum_srs = (sum_srs > SMAX) ? SMAX : (sum_srs < -SMIN) ? -SMIN : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif // INT8_ACT

#else // Vector

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
void conv2dk1_i8_vector(int8_t *input, int8_t *kernels, int8_t *output,
                        const int32_t runtime_input_width,
                        const int32_t runtime_input_channels,
                        const int32_t runtime_output_channels,
                        const int scale) {
  const int32_t input_width = CONV_INPUT_WIDTH;
  const int32_t input_channels = CONV_INPUT_CHANNELS;
  const int32_t output_channels = CONV_OUTPUT_CHANNELS;
  event0();

  constexpr int NUM_ACC = CONV2DK1_I8_NUM_ACC; // Number of accumulators
  constexpr int MMUL_M =
      CONV2DK1_I8_MMUL_M; // Matrix A M size in MxK (Input width)
  constexpr int MMUL_K = 8;
  constexpr int MMUL_N = 8;
  constexpr int CHANNEL_FACTOR = MMUL_K;
  constexpr int MMUL_MK = MMUL_M * MMUL_K;
  constexpr int MMUL_KN = MMUL_K * MMUL_N;
  constexpr int MMUL_MN = MMUL_M * MMUL_N;

  using MMUL8x8x8 = aie::mmul<MMUL_M, MMUL_K, MMUL_N, int8, int8>;
  ::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
  ::aie::set_rounding(CONV2DK1_I8_ROUNDING);

  int8_t *restrict out_ptr = output;

  const int scaleT = scale;

  // Every loop over acc_tmp is unrolled so the index is a constant: with a
  // running index the accumulator array lives in memory and the mac loop
  // spills round-trips through it, which costs both II and .text.
  MMUL8x8x8 acc_tmp[NUM_ACC];
  AIE_LOOP_UNROLL_FULL
  for (int x = 0; x < NUM_ACC; x++) {
    acc_tmp[x] = aie::zeros<acc32, MMUL_MN>();
  }

  // TODO: keeping this variable produces incorrect results and a worse
  // schedule.
  const int iw = input_width;
  const int iw_partial = (input_width / MMUL_M) / NUM_ACC;

  // constexpr int iw_partial_rem = (input_width / MMUL_M) % NUM_ACC;
  // const int iw_partial_rem = (32 / MMUL_M) % NUM_ACC;
  assert((input_width / MMUL_M) % NUM_ACC == 0);
  const int iw_partial_rem = 0; // TODO - See restriction

  assert((input_channels / CHANNEL_FACTOR) > 2); // Assume IC >= 16

  int8_t *input_begin_ptr = input;
  int8_t *input_rem_begin_ptr =
      input + iw_partial * MMUL_M * NUM_ACC * CHANNEL_FACTOR;

  if (iw_partial > 0) {

    for (int oc = 0; oc < (output_channels / CHANNEL_FACTOR); oc++) {
      for (int iw_partialc = 0; iw_partialc < iw_partial; iw_partialc++) {
#if AIE_TUNED_AIE2
        // Two pointers, one per half block, so the loads can dual-issue.
        // Rolled: LLVM's default unroll by two pipelines worse.
        int8_t *restrict in1 = input + MMUL_MK * NUM_ACC / 2;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_NO_UNROLL
        for (int ic = 0; ic < (input_channels / CHANNEL_FACTOR); ic++) {
          aie::vector<int8, MMUL_KN> in_b = aie::load_v<MMUL_KN>(kernels);
          kernels += MMUL_KN;

          AIE_LOOP_UNROLL_FULL
          for (int x = 0; x < NUM_ACC / 2; x++) {
            acc_tmp[x].mac(aie::load_v<MMUL_MK>(input + x * MMUL_MK), in_b);
            acc_tmp[x + NUM_ACC / 2].mac(
                aie::load_v<MMUL_MK>(in1 + x * MMUL_MK), in_b);
          }
          input += iw * CHANNEL_FACTOR;
          in1 += iw * CHANNEL_FACTOR;
        }
#else
        AIE_PREPARE_FOR_PIPELINING
        for (int ic = 0; ic < (input_channels / CHANNEL_FACTOR); ic++) {
          aie::vector<int8, MMUL_KN> in_b = aie::load_v<MMUL_KN>(kernels);
          kernels += MMUL_KN; // wts ic0..7(oc0..7)

          AIE_LOOP_UNROLL_FULL
          for (int x = 0; x < NUM_ACC; x++) {
            aie::vector<int8, MMUL_MK> in_a = aie::load_v<MMUL_MK>(input);
            input += MMUL_MK; // act oc0..3(ic0..7)
            acc_tmp[x].mac(in_a, in_b);
          }
          // Move to next ic/8 position but in the same input range
          input += (iw * CHANNEL_FACTOR) - MMUL_MK * NUM_ACC;
        }
#endif
        // input ptr just moves to next section

        AIE_LOOP_UNROLL_FULL
        for (int xx = 0; xx < NUM_ACC; xx++) {
          aie::vector<int8, MMUL_MN> o1 = acc_tmp[xx].to_vector<int8>(scaleT);
          aie::store_v(out_ptr, o1);
          out_ptr += MMUL_MN;
          acc_tmp[xx] = aie::zeros<acc32, MMUL_MN>();
        }
        // reset to next set of 64*NUM_ACC inputs
        input -= (input_channels * iw) - MMUL_MK * NUM_ACC;
        // reset kernel back to beginning of ic/8
        kernels -= (input_channels / CHANNEL_FACTOR) * MMUL_KN;
      }
      input = input_begin_ptr; // reset beginning of input ptr
      kernels += (input_channels / CHANNEL_FACTOR) *
                 MMUL_KN; // move to next oc/8 weights
      out_ptr +=
          (iw_partial_rem *
           MMUL_MN); // move to next oc/8 (skip remainder section if present)
    }

  } // if(iw_partial > 0) {

  if (iw_partial_rem > 0) {

    const int ocs = output_channels;
    const int ics = input_channels;

    for (int oc = 0; oc < (ocs / CHANNEL_FACTOR); oc++) {
      AIE_PREPARE_FOR_PIPELINING
      for (int ic = 0; ic < (ics / CHANNEL_FACTOR); ic++) {
        aie::vector<int8, MMUL_KN> in_b = aie::load_v<MMUL_KN>(kernels);
        kernels += MMUL_KN; // wts ic0..7(oc0..7)

        for (int x = 0; x < iw_partial_rem; x++) {
          aie::vector<int8, MMUL_MK> in_a = aie::load_v<MMUL_MK>(input);
          input += MMUL_MK; // act oc0..3(ic0..7)
          acc_tmp[x].mac(in_a, in_b);
        }
        input += (iw * CHANNEL_FACTOR) -
                 (MMUL_MK * iw_partial_rem); // Move to next ic/8 position
      }
      // input ptr just moves to next section
      for (int xx = 0; xx < iw_partial_rem; xx++) {
        aie::vector<int8, MMUL_MN> o1 = acc_tmp[xx].to_vector<int8>(scaleT);
        aie::store_v(out_ptr, o1);
        out_ptr += MMUL_MN;
        acc_tmp[xx] = aie::zeros<acc32, MMUL_MN>();
      }
      // input   -= ((ics-1)/8)*(iw*8)+(iw_partial_rem*32); // reset to
      // beginning of input ptr for remainder reset to beginning of input ptr
      // for remainder
      input = input_rem_begin_ptr;
      // kernel ptr already at next oc/8
      out_ptr +=
          (iw * CHANNEL_FACTOR) -
          (iw_partial_rem *
           MMUL_MN); // move to next oc/8 (skip remainder section if present)
    }

  } // if(iw_partial_rem > 0)

  event1();
}

#ifdef CONV_BIAS
//*****************************************************************************
// conv2d 1x1 - vector, with a bias
// act: int8, wts: int8, bias: int32 per output channel, out: int8
//
// ONNX QLinearConv: the bias, in accumulator units, seeds the accumulator,
// and an optional 256-entry table maps each int8 result (a quantized SiLU).
//*****************************************************************************
template <int NumAcc, int M>
static inline __attribute__((always_inline)) void conv2dk1_i8_bias_block(
    const int8_t *__restrict input, const int8_t *__restrict kernels,
    aie::vector<int32, 8> bias, int8_t *__restrict output, const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int input_channels = CONV_INPUT_CHANNELS;
  using MMUL = aie::mmul<M, 8, 8, int8, int8>;

  aie::vector<int32, M * 8> seed;
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < M; p++)
    seed.insert(p, bias);
  aie::accum<acc32, M * 8> seeded;
  seeded.from_vector(seed, 0);

  MMUL acc[NumAcc];
  AIE_LOOP_UNROLL_FULL
  for (int x = 0; x < NumAcc; x++)
    acc[x] = seeded;

  AIE_PREPARE_FOR_PIPELINING
  for (int ic = 0; ic < input_channels / 8; ic++) {
    aie::vector<int8, 64> b = aie::load_v<64>(kernels + ic * 64);
    AIE_LOOP_UNROLL_FULL
    for (int x = 0; x < NumAcc; x++)
      acc[x].mac(aie::load_v<M * 8>(input + ic * iw * 8 + x * M * 8), b);
  }

  AIE_LOOP_UNROLL_FULL
  for (int x = 0; x < NumAcc; x++)
    aie::store_v(output + x * M * 8, acc[x].template to_vector<int8>(scale));
}

template <bool Table>
static void
conv2dk1_i8_bias_vector(int8_t *__restrict input, int8_t *__restrict kernels,
                        int32_t *__restrict bias, int8_t *__restrict table,
                        int8_t *__restrict output, const int scale) {
  constexpr int iw = CONV_INPUT_WIDTH;
  constexpr int input_channels = CONV_INPUT_CHANNELS;
  constexpr int output_channels = CONV_OUTPUT_CHANNELS;
  constexpr int NUM_ACC = CONV2DK1_I8_NUM_ACC;
  // A row of 8k + 4 pixels leaves every other channel group's row off the
  // 64-byte alignment of an 8-pixel load, so it goes 4 pixels at a time.
  constexpr int M = iw % 8 == 0 ? CONV2DK1_I8_MMUL_M : 4;
  constexpr int blocks = iw / (NUM_ACC * M);
  constexpr int singles = iw % (NUM_ACC * M) / M;
  static_assert(iw % 4 == 0, "conv2dk1_i8 with a bias needs a width of 4k");
  static_assert(input_channels % 8 == 0 && output_channels % 8 == 0,
                "conv2dk1_i8 channels come in groups of 8");
  event0();

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(CONV2DK1_I8_ROUNDING);

  for (int oc = 0; oc < output_channels / 8; oc++) {
    const int8_t *w = kernels + oc * (input_channels / 8) * 64;
    aie::vector<int32, 8> b = aie::load_v<8>(bias + oc * 8);
    int8_t *out = output + oc * iw * 8;
    for (int x = 0; x < blocks; x++)
      conv2dk1_i8_bias_block<NUM_ACC, M>(input + x * NUM_ACC * M * 8, w, b,
                                         out + x * NUM_ACC * M * 8, scale);
    constexpr int done = blocks * NUM_ACC * M;
    for (int x = 0; x < singles; x++)
      conv2dk1_i8_bias_block<1, M>(input + (done + x * M) * 8, w, b,
                                   out + (done + x * M) * 8, scale);
  }
  if constexpr (Table)
    map_table_i8<output_channels * iw>(output, table);
  event1();
}
#endif // CONV_BIAS
#endif // INT8_ACT
#endif // Vector

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, int8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}
#endif // INT8_ACT
#else  // Vector

#ifdef INT8_ACT

#if defined(CONV_BIAS) && defined(CONV_TABLE)
void CONV2DK1_I8_SYMBOL(int8_t *input, int8_t *kernels, int32_t *bias,
                        int8_t *table, int8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  conv2dk1_i8_bias_vector<true>(input, kernels, bias, table, output, scale);
}
#elif defined(CONV_BIAS)
void CONV2DK1_I8_SYMBOL(int8_t *input, int8_t *kernels, int32_t *bias,
                        int8_t *output, const int32_t input_width,
                        const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  conv2dk1_i8_bias_vector<false>(input, kernels, bias, nullptr, output, scale);
}
#else
void CONV2DK1_I8_SYMBOL(int8_t *input, int8_t *kernels, int8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  conv2dk1_i8_vector(input, kernels, output, input_width, input_channels,
                     output_channels, scale);
}
#endif
#endif // INT8_ACT
#endif // Vector
} // extern "C"
