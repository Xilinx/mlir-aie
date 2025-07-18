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
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    uint16_t raw = (uint16_t)i;
    bufIn1[i] = *(std::bfloat16_t *)(&raw);
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

inline float custom_sqrtf(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float guess = x;
  // 5 iterations are enough for a reasonable precision
  for (int i = 0; i < 5; i++) {
    guess = 0.5f * (guess + x / guess);
  }
  return guess;
}
// Functional correctness verifyer for layer normalization.
// Signature now matches the expected form: (bufIn1, bufOut, in_elements,
// out_elements) It uses fixed matrix dimensions: 16 rows x 64 cols.
int verify_layernorm_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                            int in_elements, int out_elements) {
  int errors = 0;
  constexpr int rows = 16;
  constexpr int cols = 64;
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f; // built-in constant
  const float beta = 0.0f;  // built-in constant
  constexpr float tol = 1e-3f;
  std::vector<float> expected(rows * cols, 0.0f);
  for (int c = 0; c < cols; c++) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    // Accumulate sum and sum of squares for each column
    for (int r = 0; r < rows; r++) {
      int idx = r * cols + c;
      float val = static_cast<float>(bufIn1[idx]);
      sum += val;
      sum_sq += val * val;
    }
    float mean = sum / float(rows);
    float variance = sum_sq / float(rows) - mean * mean;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);

    // Compute expected output for the current column
    for (int r = 0; r < rows; r++) {
      int idx = r * cols + c;
      float val = static_cast<float>(bufIn1[idx]);
      float norm = (val - mean) * inv_std;
      float scaled = norm * gamma;
      float out_val = scaled + beta;
      expected[idx] = out_val;
    }
  }
  // Now compare the expected results with the computed results in bufOut
  for (int i = 0; i < (rows * cols); i++) {
    float expected_val = expected[i];
    float hw_val = static_cast<float>(bufOut[i]);
    if (std::abs(expected_val - hw_val) > tol) {
      std::cout << "Mismatch at index " << i << ": expected " << expected_val
                << ", got " << hw_val << std::endl;
      errors++;
    }
  }
  if (errors == 0)
    std::cout << "LayerNorm Passed." << std::endl;
  else
    std::cout << "LayerNorm FAILED with " << errors << " errors out of "
              << (rows * cols) << " elements." << std::endl;

  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = IN_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = IN_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_layernorm_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}