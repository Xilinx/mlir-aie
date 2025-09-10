//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/experimental/xrt_kernel.h" // for xrt::runlist
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("add_one_two");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel0 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  std::cout << "Name: " << name << std::endl;
                                  return name == "ADDONE";
                                });
  auto kernelName0 = xkernel0.get_name();
  auto xkernel1 = *std::find_if(xkernels.begin(), xkernels.end(),
                                [](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  std::cout << "Name: " << name << std::endl;
                                  return name == "ADDTWO";
                                });
  auto kernelName1 = xkernel1.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernels: " << kernelName0 << " and "
              << kernelName1 << "\n";

  auto kernel0 = xrt::kernel(context, kernelName0);

  auto bo0_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel0.group_id(1));
  auto bo0_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(3));
  auto bo0_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(4));

  auto kernel1 = xrt::kernel(context, kernelName1);

  auto bo1_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
  auto bo1_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
  auto bo1_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint32_t *bufInA = bo0_inA.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  void *bufInstr = bo0_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo0_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo0_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel 0.\n";

  unsigned int opcode = 3;
  auto run0 = kernel0(opcode, bo0_instr, instr_v.size(), bo0_inA, bo0_out);

  // same instructions as kernel1
  bufInstr = bo1_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo1_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel 1.\n";
  // Use the output of kernel0 as input to kernel1
  auto run1 = kernel1(opcode, bo1_instr, instr_v.size(), bo0_out, bo1_out);

  // Creating a runlist to contain two seperate runs
  xrt::runlist runlist = xrt::runlist(context);
  runlist.add(run0);
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  // Synchronize the output buffers to read back the results
  bo0_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo1_out.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < 64; i++) {
    uint32_t ref = (i + 1) + 1 + 2;
    if (*(bufOut + i) != ref) {
      std::cout << "Error in output " << *(bufOut + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut + i) << " == " << ref
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
