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

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")("verbosity,v",
                               po::value<int>()->default_value(0),
                               "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  test_utils::check_arg_file_exists(vm, "xclbin");
  test_utils::check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

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
  auto bo0_inB = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(4));
  auto bo0_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel0.group_id(5));

  auto kernel1 = xrt::kernel(context, kernelName1);

  auto bo1_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
  auto bo1_inA = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
  auto bo1_inB = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));
  auto bo1_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));

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
  auto run0 =
      kernel0(opcode, bo0_instr, instr_v.size(), bo0_inA, bo0_inB, bo0_out);
  run0.wait();

  bo0_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo0_out.map<uint32_t *>();

  // same instructions as kernel1
  bufInstr = bo1_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo1_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // copy kernel0 output to kernel1 input
  bufInA = bo1_inA.map<uint32_t *>();
  memcpy(bufInA, bufOut, IN_SIZE * sizeof(uint32_t));
  bo1_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel 1.\n";
  auto run1 =
      kernel1(opcode, bo1_instr, instr_v.size(), bo1_inA, bo1_inB, bo1_out);
  run1.wait();

  bo1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bufOut = bo1_out.map<uint32_t *>();

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
