//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Chaining Channels Test");
  cxxopts::ParseResult vm;

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance MLIR_AIE)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the command processor",
      cxxopts::value<std::string>())(
      "length,l", "the length of the transfer in bytes",
      cxxopts::value<int>()->default_value("1024"))(
      "verify", "enable verification path (0 or 1)",
      cxxopts::value<int>()->default_value("0"));

  try {
    vm = options.parse(argc, argv);

    if (vm.count("help")) {
      std::cout << options.help() << std::endl;
      return 1;
    }

    // Check required options
    if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << std::endl;
      return 1;
    }
  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

  int verify = vm["verify"].as<int>();

  int N = vm["length"].as<int>();
  if ((N % 4)) {
    std::cerr << "Length must be a multiple of 4 bytes." << std::endl;
    return 1;
  }

  int N_int32 = N / 4; // Convert bytes to int32 elements (1KB = 256 int32)
  int N_read = N * 4; // Read buffer is 4KB
  int N_read_int32 = N_read / 4; // 4KB = 1024 int32 elements

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
  auto bo_A = xrt::bo(device, N, XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3)); // 1KB write buffer
  auto bo_B = xrt::bo(device, N_read, XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4)); // 4KB read buffer

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer A with hex value 0xDEADBEEF (will be overwritten)
  uint32_t *bufA = bo_A.map<uint32_t *>();
  for (int i = 0; i < N_int32; i++) {
    bufA[i] = 0xDEADBEEF;
  }

  // Initialize buffer B with increasing values
  uint32_t *bufB = bo_B.map<uint32_t *>();
  for (int i = 0; i < N_read_int32; i++) {
    bufB[i] = i;
  }

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;

  int errors = 0;

  if (verify) {
    // Create buffer C for verification
    auto bo_C = xrt::bo(device, N_read, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5)); // 4KB verification buffer
    uint32_t *bufC = bo_C.map<uint32_t *>();
    for (int i = 0; i < N_read_int32; i++) {
      bufC[i] = 0xCAFEBABE; // Initialize with pattern
    }
    bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (verbosity >= 1)
      std::cout << "Running Kernel with verification." << std::endl;
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_A, bo_B, bo_C);
    run.wait();

    bo_A.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bufA = bo_A.map<uint32_t *>();
    bufC = bo_C.map<uint32_t *>();

    // Verify that buffer A was written with the initialized pattern (1 to N)
    if (verbosity >= 1)
      std::cout << "Verifying buffer A (write buffer)..." << std::endl;
    for (uint32_t i = 0; i < N_int32; i++) {
      uint32_t ref = (i + 1);
      if (bufA[i] != ref) {
        if (errors < 10) {
          std::cout << "Error in buffer A at index " << i << ": expected " << ref
                    << ", got " << bufA[i] << std::endl;
        }
        errors++;
      }
    }

    // Verify buffer C contains the data read from B
    if (verbosity >= 1)
      std::cout << "Verifying buffer C (verification buffer)..." << std::endl;
    for (uint32_t i = 0; i < N_read_int32; i++) {
      uint32_t ref = i; // Buffer B was initialized with increasing values
      if (bufC[i] != ref) {
        if (errors < 10) {
          std::cout << "Error in buffer C at index " << i << ": expected " << ref
                    << ", got " << bufC[i] << std::endl;
        }
        errors++;
      }
    }
  } else {
    if (verbosity >= 1)
      std::cout << "Running Kernel without verification." << std::endl;
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_A, bo_B);
    run.wait();

    bo_A.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bufA = bo_A.map<uint32_t *>();

    // Verify that buffer A was written with the initialized pattern (1 to N)
    if (verbosity >= 1)
      std::cout << "Verifying buffer A (write buffer)..." << std::endl;
    for (uint32_t i = 0; i < N_int32; i++) {
      uint32_t ref = (i + 1);
      if (bufA[i] != ref) {
        if (errors < N) {
          std::cout << "Error in buffer A at index " << i << ": expected " << ref
                    << ", got " << bufA[i] << std::endl;
        }
        errors++;
      }
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
