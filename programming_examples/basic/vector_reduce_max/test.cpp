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
#include <cstdint>
#include <stdfloat>
#ifndef DTYPE
#define DTYPE std::bfloat16_t
#endif
// ------------------------------------------------------
// Configure this to match your buffer data type
// -----------------------------------------------------
using DATATYPE = DTYPE;

void initialize_bufIn1(DATATYPE *bufIn1, int SIZE) {
  DATATYPE max = std::numeric_limits<DATATYPE>::lowest();
  for (int i = 0; i < SIZE; i++) {
    DATATYPE next;
    if constexpr (std::is_same_v<DATATYPE, std::bfloat16_t> &&
                  std::is_same_v<DATATYPE, std::bfloat16_t>) {
      next = test_utils::random_bfloat16_t((std::bfloat16_t)-4.0,
                                           (std::bfloat16_t)8.0);
    } else if constexpr (std::is_same_v<DATATYPE, int32_t> &&
                         std::is_same_v<DATATYPE, int32_t>) {
      next = test_utils::random_int32_t(100000);
    } else {
      std::cerr << "Unsupported data type" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (next > max)
      max = next;
    bufIn1[i] = next;
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE *bufOut, int SIZE) { memset(bufOut, 0, SIZE); }

// Functional correctness verifyer
int verify_vector_reduce_max(DATATYPE *bufIn1, DATATYPE *bufOut, int SIZE,
                             int verbosity) {
  int errors = 0;

  // Calculate max within the function
  DATATYPE max = std::numeric_limits<DATATYPE>::lowest();
  for (int i = 0; i < SIZE; i++) {
    if (bufIn1[i] > max)
      max = bufIn1[i];
  }

  if (bufOut[0] != max) {
    errors++;
    std::cout << "max is " << max << " calc " << bufOut[0] << std::endl;
  } else {
    if (verbosity >= 1)
      std::cout << "max is " << max << " calc " << bufOut[0] << std::endl;
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE, DATATYPE, initialize_bufIn1,
                              initialize_bufOut, verify_vector_reduce_max>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}