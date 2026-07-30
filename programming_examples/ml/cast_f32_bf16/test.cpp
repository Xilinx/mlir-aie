//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = float;
using DATATYPE_OUT = test_utils::bfloat16_t;
#endif

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_volume) {
  for (int i = 0; i < in_volume; i++) {
    bufIn1[i] = 8.0f * (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f);
  }
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int out_volume) {
  memset(bufOut, 0, out_volume * sizeof(DATATYPE_OUT));
}

int verify_cast_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                       int in_elements, int out_elements) {
  int errors = 0, pass = 0;
  for (int i = 0; i < in_elements; i++) {
    // round-to-nearest-even, same rule as the kernel's aie::conv_even.
    DATATYPE_OUT expected = test_utils::bfloat16_from_float(bufIn1[i]);
    DATATYPE_OUT got = bufOut[i];
    if (expected != got) {
      std::cout << "Mismatch at index " << i << ": expected "
                << test_utils::bfloat16_to_float(expected) << ", got "
                << test_utils::bfloat16_to_float(got) << std::endl;
      errors++;
    } else {
      pass++;
    }
  }

  if (errors == 0)
    std::cout << "cast_f32_bf16 Passed : " << pass << " out of " << in_elements
              << std::endl;
  else
    std::cout << "cast_f32_bf16 FAILED with " << errors << " errors out of "
              << in_elements << " elements." << std::endl;
  return errors;
}

int main(int argc, const char *argv[]) {
  args myargs = parse_args(argc, argv);
  int in_volume = (ROWS * COLS);
  int out_volume = in_volume;
  return setup_and_run_aie(
      verify_cast_kernel,
      std::make_tuple(make_in<DATATYPE_IN1>(in_volume, initialize_bufIn1)),
      make_out<DATATYPE_OUT>(out_volume, initialize_bufOut), myargs);
}
