//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_IN2 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = i + 1;
}

// Initialize Input buffer 2
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
  bufIn2[0] = 3; // scaleFactor
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifier
int verify_vector_scalar_mul(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
  int errors = 0;

  for (int i = 0; i < SIZE; i++) {
    int32_t ref = bufIn1[i] * bufIn2[0];
    int32_t test = bufOut[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int IN2_VOLUME = IN2_SIZE / sizeof(DATATYPE_IN2);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_IN2, DATATYPE_OUT,
                              initialize_bufIn1, initialize_bufIn2,
                              initialize_bufOut, verify_vector_scalar_mul>(
      IN1_VOLUME, IN2_VOLUME, OUT_VOLUME, myargs);
  return res;
}
