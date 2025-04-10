//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

//*****************************************************************************
// Modify this section to customize buffer datatypes, initialization functions,
// and verify function. The other place to reconfigure your design is the
// Makefile.
//*****************************************************************************

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE_IN1 = std::bfloat16_t;
using DATATYPE_IN2 = std::bfloat16_t;
using DATATYPE_OUT = std::bfloat16_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = test_utils::random_bfloat16_t((std::bfloat16_t)1.0,
                                              (std::bfloat16_t)-0.5);
}

// Initialize Input buffer 2
void initialize_bufIn2(DATATYPE_IN2 *bufIn2, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn2[i] = test_utils::random_bfloat16_t((std::bfloat16_t)1.0,
                                              (std::bfloat16_t)-0.5);                                              
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify_vector_scalar_mul(DATATYPE_IN1 *bufIn1, DATATYPE_IN2 *bufIn2,
                             DATATYPE_OUT *bufOut, int SIZE, int verbosity) {
  int errors = 0;

  for (int i = 0; i < SIZE; i++) {
    DATATYPE_OUT ref = bufIn1[i] + bufIn2[i];
    DATATYPE_OUT test = bufOut[i];
    if (!test_utils::nearly_equal(ref, test, 0.00390625)) {
      if (verbosity >= 1)
        std::cout << "Error in output " << i << ": " << test << " != " << ref << " from "
                  << bufIn1[i] << " + " << bufIn2[i] << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << i << ": " << test << " == " << ref << std::endl;
    }
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

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
