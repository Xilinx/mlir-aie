//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "test_utils.h"
#include "xrt/xrt_bo.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE = std::uint8_t;
#endif

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  std::cout << std::endl << "Running...";

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, PASSTHROUGH_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, PASSTHROUGH_SIZE * sizeof(DATATYPE) + trace_size,
              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  for (int i = 0; i < PASSTHROUGH_SIZE; i++)
    bufInA[i] = i;

  // Zero out buffer bo_out
  DATATYPE *bufOut = bo_out.map<DATATYPE *>();
  memset(bufOut, 0, PASSTHROUGH_SIZE * sizeof(DATATYPE) + trace_size);

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_out);
  run.wait();

  // Sync device to host memories
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Compare out to in
  int errors = 0;
  for (int i = 0; i < PASSTHROUGH_SIZE; i++) {
    if (bufOut[i] != bufInA[i])
      errors++;
  }

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufOut) +
                                    (PASSTHROUGH_SIZE * sizeof(DATATYPE)),
                                trace_size, vm["trace_file"].as<std::string>());
  }

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
