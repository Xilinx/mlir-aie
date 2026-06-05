//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = test_utils::bfloat16_t;
using DATATYPE_OUT = test_utils::bfloat16_t;
#endif

// Op selector — populated from the `NORM_OP` env var so the linked-in
// verify function picks the right reference without dragging another
// argv plumbing layer through the xrt_test_wrapper template.
static std::string norm_op() {
  const char *env = std::getenv("NORM_OP");
  return env ? std::string(env) : std::string("rms");
}

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_volume) {
  for (int i = 0; i < in_volume; i++) {
    bufIn1[i] =
        test_utils::random_bfloat16_t(test_utils::bfloat16_from_float(8.0f),
                                      test_utils::bfloat16_from_float(-4.0f));
  }
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int out_volume) {
  memset(bufOut, 0, out_volume);
}

int verify_norm_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                       int in_elements, int out_elements) {
  const std::string op = norm_op();
  if (op != "rms" && op != "layer") {
    std::cerr << "NORM_OP must be 'rms' or 'layer' (got '" << op << "')\n";
    return 1;
  }

  constexpr float epsilon = 1e-5f;
  constexpr float gamma = 1.0f;
  constexpr float beta = 0.0f;
  const float tol = (op == "layer") ? 0.1f : 0.05f;

  std::vector<float> expected(ROWS * COLS, 0.0f);
  for (int r = 0; r < ROWS; r++) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int c = 0; c < COLS; c++) {
      float val = test_utils::bfloat16_to_float(bufIn1[r * COLS + c]);
      sum += val;
      sum_sq += val * val;
    }
    if (op == "rms") {
      float rms = std::sqrt(sum_sq / float(COLS) + epsilon);
      for (int c = 0; c < COLS; c++) {
        int idx = r * COLS + c;
        float val = test_utils::bfloat16_to_float(bufIn1[idx]);
        expected[idx] = (val * gamma) / rms;
      }
    } else { // layer
      float mean = sum / float(COLS);
      float variance = sum_sq / float(COLS) - mean * mean;
      float inv_std = 1.0f / std::sqrt(variance + epsilon);
      for (int c = 0; c < COLS; c++) {
        int idx = r * COLS + c;
        float val = test_utils::bfloat16_to_float(bufIn1[idx]);
        expected[idx] = (val - mean) * inv_std * gamma + beta;
      }
    }
  }

  int errors = 0, pass = 0;
  for (int i = 0; i < (ROWS * COLS); i++) {
    float ev = expected[i];
    float hv = test_utils::bfloat16_to_float(bufOut[i]);
    float diff = std::abs(ev - hv);
    if (diff > tol) {
      std::cout << "Mismatch at index " << i << ": expected " << ev << ", got "
                << hv << ", diff = " << diff << std::endl;
      errors++;
    } else {
      pass++;
    }
  }

  const std::string label = (op == "rms") ? "RMSNorm" : "LayerNorm";
  if (errors == 0)
    std::cout << label << " Passed : " << pass << " out of " << (ROWS * COLS)
              << std::endl;
  else
    std::cout << label << " FAILED with " << errors << " errors out of "
              << (ROWS * COLS) << " elements." << std::endl;
  return errors;
}

int main(int argc, const char *argv[]) {
  args myargs = parse_args(argc, argv);
  int in_volume = (ROWS * COLS);
  int out_volume = in_volume;
  return setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                           initialize_bufOut, verify_norm_kernel>(
      in_volume, out_volume, myargs);
}
