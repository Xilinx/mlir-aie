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

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

// relu reference implementation
std::bfloat16_t relu_bf16(std::bfloat16_t &input) {
  // Return the relu output
  return (input > std::bfloat16_t(0.0f)) ? input : std::bfloat16_t(0.0f);
}

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Passthrough DMAs Test");
  cxxopts::ParseResult vm;

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "length,l", "the length of the transfer in std::bfloat16_t",
      cxxopts::value<int>()->default_value("4096"));

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

  int verbosity = vm["verbosity"].as<int>();

  int N = vm["length"].as<int>();
  if ((N % 1024)) {
    std::cerr << "Length must be a multiple of 1024." << std::endl;
    return 1;
  }

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

  xrt::elf elf(vm["instr"].as<std::string>());
  xrt::module mod{elf};

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::ext::kernel(context, mod, kernelName);

  xrt::bo bo_inA = xrt::ext::bo{device, N * sizeof(std::bfloat16_t)};
  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(std::bfloat16_t)};

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  std::bfloat16_t *bufInA = bo_inA.map<std::bfloat16_t *>();
  std::vector<std::bfloat16_t> srcVecA;
  for (int i = 0; i < N; i++)
    srcVecA.push_back(std::bfloat16_t(i * 0.05f + -3.0f)); // Example data
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(std::bfloat16_t)));

  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  // Setup run to configure
  auto cfg_run = kernel(opcode, 0, 0, bo_inA, bo_out);
  cfg_run.wait();
  auto start = std::chrono::high_resolution_clock::now();
  // Test run
  auto run = kernel(opcode, 0, 0, bo_inA, bo_out);
  run.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  const float npu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::cout << std::endl;
  std::cout << "Latency (us): " << npu_time << std::endl;
  std::cout << std::endl;

  double total_bytes = 2.0 * N * sizeof(std::bfloat16_t); // input and output
  double bandwidth_GBps = total_bytes / (npu_time * 1e-6) / 1e9;
  std::cout << "Effective Bandwidth: " << bandwidth_GBps << " GB/s"
            << std::endl;

  std::bfloat16_t *bufOut = bo_out.map<std::bfloat16_t *>();

  int errors = 0;

  for (int i = 0; i < N; i++) {
    std::bfloat16_t ref = relu_bf16(srcVecA[i]);
    if (!test_utils::nearly_equal(*(bufOut + i), ref)) {
      errors++;
      // Print the first 100 mismatches
      if (errors <= 100) {
        std::cout << "Mismatch at index " << i << ": "
                  << "Expected: " << ref << ", "
                  << "Got: " << *(bufOut + i) << std::endl;
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
