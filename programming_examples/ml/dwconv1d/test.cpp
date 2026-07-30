//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = test_utils::bfloat16_t; // x, already 'same'-padded
using DATATYPE_IN2 =
    test_utils::bfloat16_t; // w = [taps(KERNEL_SIZE) | bias-or-pad(1)]
using DATATYPE_OUT = test_utils::bfloat16_t;
#endif

// Matches dwconv1d.py / dwconv1d.cc: the aligned-load kernel's fixed tail
// slack past the halo, independent of KERNEL_SIZE (see dwconv1d.cc header).
static constexpr int kTailSlack = 16;
static constexpr int kPad = (KERNEL_SIZE - 1) / 2;
static constexpr int kInRow = SEQ_LEN + kTailSlack;
// Always KERNEL_SIZE + 1 (bias tap, or an unused zero-padding column when
// BIAS=0): aie.dma_bd needs a 4-byte-aligned transfer length and
// KERNEL_SIZE alone is odd, see the w_row comment in dwconv1d.py.
static constexpr int kWRow = KERNEL_SIZE + 1;

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_volume) {
  // Zero everything first: the leading/trailing 'same'-padding halo and the
  // don't-care tail slack the kernel's aligned loads read past but never
  // route to the output (see dwconv1d.cc).
  memset(bufIn1, 0, in_volume * sizeof(DATATYPE_IN1));
  for (int r = 0; r < CHANNELS; r++) {
    DATATYPE_IN1 *row = bufIn1 + (size_t)r * kInRow;
    for (int t = 0; t < SEQ_LEN; t++)
      row[kPad + t] = test_utils::random_bfloat16_t(
          test_utils::bfloat16_from_float(4.0f),
          test_utils::bfloat16_from_float(-2.0f)); // [-2, 2)
  }
}

void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int w_volume) {
  memset(bufIn2, 0, w_volume * sizeof(DATATYPE_IN2));
  for (int r = 0; r < CHANNELS; r++) {
    DATATYPE_IN2 *wrow = bufIn2 + (size_t)r * kWRow;
    for (int p = 0; p < KERNEL_SIZE; p++)
      wrow[p] = test_utils::random_bfloat16_t(
          test_utils::bfloat16_from_float(2.0f),
          test_utils::bfloat16_from_float(-1.0f)); // [-1, 1)
#if BIAS
    wrow[KERNEL_SIZE] = test_utils::random_bfloat16_t(
        test_utils::bfloat16_from_float(2.0f),
        test_utils::bfloat16_from_float(-1.0f)); // [-1, 1)
#endif
    // wrow[KERNEL_SIZE] stays 0 (memset above) when BIAS=0, the unused
    // alignment padding column dwconv1d.cc never reads.
  }
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int out_volume) {
  memset(bufOut, 0, out_volume * sizeof(DATATYPE_OUT));
}

// setup_and_run_aie's verify contract: verify(in0*, in1*, ..., out*,
// first_input_volume, verbosity) -> error count. first_input_volume is x's
// element count (CHANNELS * kInRow); w's/out's layout come from the
// CHANNELS/SEQ_LEN/KERNEL_SIZE/BIAS macros the same way the other buffers
// are built.
int verify_dwconv1d_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                           DATATYPE_OUT *bufOut, int in_volume, int verbosity) {
  int errors = 0, pass = 0;

  for (int r = 0; r < CHANNELS; r++) {
    const DATATYPE_IN1 *row = bufIn1 + (size_t)r * kInRow;
    const DATATYPE_IN2 *wrow = bufIn2 + (size_t)r * kWRow;

    for (int t = 0; t < SEQ_LEN; t++) {
      double acc =
          BIAS ? (double)test_utils::bfloat16_to_float(wrow[KERNEL_SIZE]) : 0.0;
      for (int p = 0; p < KERNEL_SIZE; p++)
        acc += (double)test_utils::bfloat16_to_float(wrow[p]) *
               (double)test_utils::bfloat16_to_float(row[t + p]);

      // Exact compare, not a tolerance: device measurement across every
      // (bias, kernel_size) combination this suite exercises is bit-exact
      // (max abs err 0), the same style cast_f32_bf16's test.cpp uses.
      DATATYPE_OUT expected = test_utils::bfloat16_from_float((float)acc);
      DATATYPE_OUT got = bufOut[(size_t)r * SEQ_LEN + t];
      if (expected != got) {
        if (errors < 10)
          std::cout << "Mismatch at channel " << r << " t " << t
                    << ": expected " << test_utils::bfloat16_to_float(expected)
                    << ", got " << test_utils::bfloat16_to_float(got)
                    << std::endl;
        errors++;
      } else {
        pass++;
      }
    }
  }

  if (errors == 0)
    std::cout << "dwconv1d Passed : " << pass << " out of "
              << (CHANNELS * SEQ_LEN) << std::endl;
  else
    std::cout << "dwconv1d FAILED with " << errors << " errors out of "
              << (CHANNELS * SEQ_LEN) << " elements." << std::endl;
  return errors;
}

int main(int argc, const char *argv[]) {
  args myargs = parse_args(argc, argv);
  int in_volume = CHANNELS * kInRow;
  int w_volume = CHANNELS * kWRow;
  int out_volume = CHANNELS * SEQ_LEN;

  return setup_and_run_aie(
      verify_dwconv1d_kernel,
      std::make_tuple(make_in<DATATYPE_IN1>(in_volume, initialize_bufIn1),
                      make_in<DATATYPE_IN2>(w_volume, initialize_bufIn2)),
      make_out<DATATYPE_OUT>(out_volume, initialize_bufOut), myargs);
}
