//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using INOUT_DATATYPE = uint8_t;
using WEIGHTS_DATATYPE = int8_t;
#endif

#include "test_utils.h"

constexpr int DEPTH = 8;
constexpr int HEIGHT = 8;
constexpr int WIDTH = 8;
constexpr int IN_CHANNELS = 8;
constexpr int OUT_CHANNELS = 8;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(argv[1]);  // instr.txt or insts.bin

  int verbosity = 1;
  std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity, argv[2],
                                   argv[3]);  // xclbin, kernel_name

  // Set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  auto input_size = DEPTH * HEIGHT * WIDTH * IN_CHANNELS;
  auto weights_size = IN_CHANNELS * OUT_CHANNELS;  // 1x1 conv for now
  auto output_size = DEPTH * HEIGHT * WIDTH * OUT_CHANNELS;

  auto bo_inA = xrt::bo(device, input_size * sizeof(INOUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weights = xrt::bo(device, weights_size * sizeof(WEIGHTS_DATATYPE),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, output_size * sizeof(INOUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize input data
  INOUT_DATATYPE *bufInA = bo_inA.map<INOUT_DATATYPE *>();
  for (int i = 0; i < input_size; i++) {
    bufInA[i] = i % 256;
  }

  // Initialize weights (all ones for simple test)
  WEIGHTS_DATATYPE *bufWeights = bo_weights.map<WEIGHTS_DATATYPE *>();
  for (int i = 0; i < weights_size; i++) {
    bufWeights[i] = 1;
  }

  // Initialize output buffer to zero
  INOUT_DATATYPE *bufOut = bo_out.map<INOUT_DATATYPE *>();
  memset(bufOut, 0, output_size * sizeof(INOUT_DATATYPE));

  // Sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel
  std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_weights, bo_out);
  run.wait();

  // Sync device to host memories
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "Kernel completed successfully!\n";

  // Verify result
  std::cout << "First 16 output values: ";
  for (int i = 0; i < 16; i++) {
    std::cout << (int)bufOut[i] << " ";
  }
  std::cout << "\n";

  std::cout << "PASS!\n";
  return 0;
}
