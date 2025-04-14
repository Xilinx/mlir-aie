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

// #include <bits/stdc++.h>
// #include <boost/program_options.hpp>
// #include <cmath>
// #include <cstdint>
// #include <fstream>
// #include <iostream>
// #include <sstream>
// #include <stdfloat>
// #include <vector>

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

// Functional correctness verifyer
int verify_relu_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut, int SIZE,
                       int verbosity) {
  int errors = 0;

  for (uint32_t i = 0; i < SIZE; i++) {
    // If the input is nan, lets just say its good
    if (std::isnan(bufIn1[i]))
      continue;

    DATATYPE_OUT ref = (DATATYPE_OUT)0;
    if (bufIn1[i] > (DATATYPE_OUT)0)
      ref = bufIn1[i];
    if (!test_utils::nearly_equal(ref, bufOut[i])) {
      std::cout << "Error in output " << bufOut[i] << " != " << ref << " from "
                << bufIn1[i] << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << bufOut[i] << " == " << ref
                  << std::endl;
    }
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
                              initialize_bufOut, verify_relu_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
