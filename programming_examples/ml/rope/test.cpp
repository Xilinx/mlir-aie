//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED

using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_IN2 = std::bfloat16_t; // For LUT (cos,sin) pairs
using DATATYPE_OUT = std::bfloat16_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_volume) {
  for (int i = 0; i < in_volume; i++) {
    DATATYPE_IN1 val = test_utils::random_bfloat16_t((std::bfloat16_t)8.0,
                                                     (std::bfloat16_t)-4.0);
    bufIn1[i] = val;
  }
}

// Initialize LUT buffer
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int lut_volume) {
  constexpr float theta = 10000.0f;
  int dims = COLS / 2;
  std::vector<float> inv_freq(dims, 0.0f);
  for (int i = 0; i < dims; i++) {
    float exponent = static_cast<float>(2 * i) / static_cast<float>(COLS);
    inv_freq[i] = 1.0f / std::pow(theta, exponent);
  }
  for (int r = 0; r < ROWS; r++) {
    for (int i = 0; i < dims; i++) {
      float angle = r * inv_freq[i];
      float cos_val = std::cos(angle);
      float sin_val = std::sin(angle);
      int base_idx = r * COLS + 2 * i;
      bufIn2[base_idx] = static_cast<DATATYPE_IN2>(cos_val);
      bufIn2[base_idx + 1] = static_cast<DATATYPE_IN2>(sin_val);
    }
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int out_volume) {
  memset(bufOut, 0, out_volume * sizeof(DATATYPE_OUT));
}

int verify_rope_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                       DATATYPE_OUT *bufOut, int in_elements,
                       int lut_elements) {
  int errors = 0;
  int pass = 0;
  int dims = COLS / 2;
  std::vector<float> expected(ROWS * COLS, 0.0f);

  for (int r = 0; r < ROWS; r++) {
    for (int i = 0; i < dims; i++) {
      int input_base_idx = r * COLS + 2 * i;
      int lut_base_idx = r * COLS + 2 * i;

      float x_even = static_cast<float>(bufIn1[input_base_idx]);
      float x_odd = static_cast<float>(bufIn1[input_base_idx + 1]);
      float cos_val = static_cast<float>(bufIn2[lut_base_idx]);
      float sin_val = static_cast<float>(bufIn2[lut_base_idx + 1]);

      expected[input_base_idx] = x_even * cos_val - x_odd * sin_val;
      expected[input_base_idx + 1] = x_even * sin_val + x_odd * cos_val;
    }
  }

  for (int i = 0; i < (ROWS * COLS); i++) {
    float expected_val = expected[i];
    float hw_val = static_cast<float>(bufOut[i]);
    if (std::abs(expected_val - hw_val) > 0.05f) {
      std::cout << "Mismatch at index " << i << ": expected " << expected_val
                << ", got " << hw_val << std::endl;
      errors++;
    } else
      pass++;
  }
  if (errors == 0)
    std::cout << "ROPE Passed : " << pass << " out of " << (ROWS * COLS)
              << std::endl;
  else
    std::cout << "ROPE FAILED with " << errors << " errors out of "
              << (ROWS * COLS) << " elements." << std::endl;

  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  args myargs = parse_args(argc, argv);

  int in_volume = (ROWS * COLS);
  int lut_volume = in_volume;
  int out_volume = in_volume;

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_IN2, DATATYPE_OUT,
                              initialize_bufIn1, initialize_bufIn2,
                              initialize_bufOut, verify_rope_kernel>(
      in_volume, lut_volume, out_volume, myargs);
  return res;
}