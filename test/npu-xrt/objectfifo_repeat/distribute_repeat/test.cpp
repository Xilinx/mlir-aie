//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("distribute_repeat");
  test_utils::add_default_options(options);
  options.add_options()("length,l", "the length of the transfer in int32_t",
                        cxxopts::value<int>()->default_value("4096"))(
      "repeat,r", "the compute tile repeat count",
      cxxopts::value<int>()->default_value("7"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  int N = vm["length"].as<int>();
  if ((N % 2 == 1)) {
    std::cerr << "Length must be a multiple of 2." << std::endl;
    return 1;
  }
  int repeat_count = vm["repeat"].as<int>();
  int out_size = N * repeat_count;

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>()
              << std::endl;
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>()
              << std::endl;
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context." << std::endl;
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_inB = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, out_size * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  int32_t *bufInA = bo_inA.map<int32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < N; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;
  for (uint32_t i = 0; i < out_size / 2; i++) {
    uint32_t ref = (i % (N / 2)) + 2;
    if (*(bufOut + i) != ref) {
      std::cout << "error at index[" << i << "]: expected " << ref << " got "
                << *(bufOut + i) << std::endl;
      errors++;
    }
  }
  for (uint32_t i = out_size / 2; i < out_size; i++) {
    uint32_t ref = (i % (N / 2)) + (N / 2) + 3;
    if (*(bufOut + i) != ref) {
      std::cout << "error at index[" << i << "]: expected " << ref << " got "
                << *(bufOut + i) << std::endl;
      errors++;
    }
  }

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
