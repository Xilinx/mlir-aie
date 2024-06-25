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

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_aie.h"
#include "experimental/xrt_kernel.h" // for xrt::runlist

#include "test_utils.h"

namespace po = boost::program_options;

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
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  constexpr int IN_SIZE = 1024;
  constexpr int OUT_SIZE = 1024;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr_0 = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA_0 = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out_0 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  auto bo_instr_1 = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA_1 = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out_1 = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initializing the input vectors 
  std::vector<uint32_t> srcVecA;
  std::vector<uint32_t> srcVecA_1;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);

  for (int i = 0; i < IN_SIZE; i++)
    srcVecA_1.push_back(i + 2);

  // Getting handles to the input data BOs and copying input data to them
  uint32_t *bufInA_0 = bo_inA_0.map<uint32_t *>();
  uint32_t *bufInA_1 = bo_inA_1.map<uint32_t *>();
  memcpy(bufInA_0, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));
  memcpy(bufInA_1, srcVecA_1.data(), (srcVecA_1.size() * sizeof(uint32_t)));

  // Getting handles to the instruction sequence BOs and copy data to them
  void *bufInstr_0 = bo_instr_0.map<void *>();
  void *bufInstr_1 = bo_instr_1.map<void *>();
  memcpy(bufInstr_0, instr_v.data(), instr_v.size() * sizeof(int));
  memcpy(bufInstr_1, instr_v.data(), instr_v.size() * sizeof(int));

  // Synchronizing BOs
  bo_instr_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  // Creating a runlist to contain two seperate runs
  xrt::runlist runlist = xrt::runlist(context);

  // Creating the first run
  xrt::run run0 = xrt::run(kernel);
  run0.set_arg(0, opcode);
  run0.set_arg(1, bo_instr_0);
  run0.set_arg(2, instr_v.size());
  run0.set_arg(3, bo_inA_0);
  run0.set_arg(4, bo_out_0);
  run0.set_arg(5, 0);
  run0.set_arg(6, 0);
  run0.set_arg(7, 0);

  // Creating the second run
  xrt::run run1 = xrt::run(kernel);
  run1.set_arg(0, opcode);
  run1.set_arg(1, bo_instr_1);
  run1.set_arg(2, instr_v.size());
  run1.set_arg(3, bo_inA_1);
  run1.set_arg(4, bo_out_1);
  run1.set_arg(5, 0);
  run1.set_arg(6, 0);
  run1.set_arg(7, 0);

  // Executing and waiting on the runlist
  runlist.add(run0);
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  // Synchronizing the output BOs
  bo_out_0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut_0 = bo_out_0.map<uint32_t *>();
  uint32_t *bufOut_1 = bo_out_1.map<uint32_t *>();

  int errors = 0;

  std::cout << "Checking run 0" << std::endl;
  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = i + 2;
    if (*(bufOut_0 + i) != ref) {
      std::cout << "Error in output " << *(bufOut_0 + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut_0 + i) << " == " << ref
                << std::endl;
    }
  }

  
  std::cout << "Checking run 1" << std::endl;
  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = i + 3;
    if (*(bufOut_1 + i) != ref) {
      std::cout << "Error in output " << *(bufOut_1 + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut_1 + i) << " == " << ref
                << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
