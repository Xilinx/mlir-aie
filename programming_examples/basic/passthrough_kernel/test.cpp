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

#ifdef USE_DYNAMIC_TXN
#include "generated_txn.h"
#endif

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

// Functional correctness verifyer
int verify_passthrough_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                              int SIZE, int verbosity) {
  int errors = 0;

  for (int i = 0; i < SIZE; i++) {
    int32_t ref = bufIn1[i];
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

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  int in1_volume = IN1_SIZE / sizeof(DATATYPE_IN1);
  int out_volume = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

#ifdef USE_DYNAMIC_TXN
  // If --dynamic-size is given, override the compiled-in buffer sizes
  // and generate TXN instructions for that size at runtime.
  // BD addresses are hardware constants for NPU2 shim tile (0,0).
  constexpr uint32_t INPUT_BD_ADDR = 118784;  // 0x1D000
  constexpr uint32_t OUTPUT_BD_ADDR = 118816; // 0x1D020
  if (myargs.dynamic_size > 0) {
    uint32_t size_bytes = myargs.dynamic_size;
    in1_volume = size_bytes / sizeof(DATATYPE_IN1);
    out_volume = size_bytes / sizeof(DATATYPE_OUT);
    myargs.generate_instr = [size_bytes]() {
      return generate_txn_sequence(size_bytes / 4, INPUT_BD_ADDR,
                                   OUTPUT_BD_ADDR);
    };
  } else {
    myargs.generate_instr = []() {
      // Default: use compile-time size
      return generate_txn_sequence(IN1_SIZE / 4, INPUT_BD_ADDR, OUTPUT_BD_ADDR);
    };
  }
#endif

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_passthrough_kernel>(
      in1_volume, out_volume, myargs);
  return res;
}
