//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "../../../runtime_lib/test_lib/xrt_test_wrapper.h"
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
#define IN1_SIZE 16384
#define OUT_SIZE 32768

using DATATYPE_IN1 = std::uint8_t;
using DATATYPE_OUT = std::uint8_t;
#endif

// Initialize Input buffer 1
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = i;
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

// Functional correctness verifyer for repeat functionality
int verify_passthrough_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                              int SIZE, int verbosity) {
  int errors = 0;
  constexpr int REPEAT_COUNT = 2;
  constexpr int CHUNK_SIZE = 1024; // Must match chunk_size in AIE design

  int num_chunks = SIZE / CHUNK_SIZE;

  // Verify chunk-based repetition: each chunk repeated REPEAT_COUNT times
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    for (int repeat = 0; repeat < REPEAT_COUNT; repeat++) {
      for (int i = 0; i < CHUNK_SIZE; i++) {
        int in_idx = chunk * CHUNK_SIZE + i;
        int out_idx = (chunk * REPEAT_COUNT + repeat) * CHUNK_SIZE + i;
        
        int32_t ref = bufIn1[in_idx];
        int32_t test = bufOut[out_idx];
        
        if (test != ref) {
          errors++;
        } else {
          if (verbosity >= 1) 
            std::cout << "Correct at chunk " << chunk << ", repeat " << repeat 
                     << ", index " << i << ": output[" << out_idx << "] = " << test 
                     << " == input[" << in_idx << "] = " << ref << std::endl;
        }
      }
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
                              initialize_bufOut, verify_passthrough_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
