//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using INOUT0_DATATYPE = std::uint32_t;
using INOUT1_DATATYPE = std::uint32_t;
#endif

namespace po = boost::program_options;

// ----------------------------------------------------------------------------
// Verify results (specific to our design example)
// ----------------------------------------------------------------------------
template <typename Tout>
int verify(int CSize, std::vector<Tout> C, int verbosity) {
  int errors = 0;
  for (uint32_t i = 0; i < CSize; i++) {
    uint32_t ref = i + 2;
    if (C[i] != ref) {
      std::cout << "Error in output " << C[i] << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << C[i] << " == " << ref << std::endl;
    }
  }
  return errors;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();

  // ------------------------------------------------------
  // Configure this to match your design's buffer size
  // ------------------------------------------------------
  int INOUT0_VOLUME = 64; // Input only, 64x uint32_t in this example
  int INOUT1_VOLUME = 64; // Output only, 64x uint32_t in this example

  size_t INOUT0_SIZE = INOUT0_VOLUME * sizeof(INOUT0_DATATYPE);
  size_t INOUT1_SIZE = INOUT1_VOLUME * sizeof(INOUT1_DATATYPE);

  size_t OUT_SIZE = INOUT1_SIZE;

  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inout0 =
      xrt::bo(device, INOUT0_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inout1 =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize Inout buffer 0
  INOUT0_DATATYPE *bufInOut0 = bo_inout0.map<INOUT0_DATATYPE *>();
  std::vector<INOUT0_DATATYPE> AVec(INOUT0_VOLUME);
  for (int i = 0; i < INOUT0_VOLUME; i++)
    AVec[i] = i + 1;
  memcpy(bufInOut0, AVec.data(), (AVec.size() * sizeof(INOUT0_DATATYPE)));

  // Initialize Inout buffer 1
  char *bufInOut1 = bo_inout1.map<char *>();
  std::vector<INOUT1_DATATYPE> CVec(INOUT1_VOLUME);
  memset(bufInOut1, 0, OUT_SIZE); // Zeroes out INOUT1_VOLUME

  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inout1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  int errors = 0;

  // ------------------------------------------------------
  // Main run
  // ------------------------------------------------------

  // Run kernel
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(bo_instr, instr_v.size(), bo_inout0, bo_inout1);
  run.wait();
  bo_inout1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Copy output results and verify they are correct
  memcpy(CVec.data(), bufInOut1, (CVec.size() * sizeof(INOUT1_DATATYPE)));
  if (do_verify) {
    if (verbosity >= 1) {
      std::cout << "Verifying results ..." << std::endl;
    }
    errors = verify(INOUT1_VOLUME, CVec, verbosity);
  } else {
    if (verbosity >= 1)
      std::cout << "WARNING: results not verified." << std::endl;
  }

  // ------------------------------------------------------
  // Print verification results
  // ------------------------------------------------------
  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
