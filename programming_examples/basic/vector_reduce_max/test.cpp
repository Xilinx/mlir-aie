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
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif

std::int32_t max = (std::int32_t)-2147483648;

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    std::int32_t next = test_utils::random_int32_t(100000);
    if (next > max)
      max = next;
    bufIn1[i] = next;
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer
int verify_vector_reduce_max(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                             int SIZE, int verbosity) {
  int errors = 0;

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

  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_vector_reduce_max>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
