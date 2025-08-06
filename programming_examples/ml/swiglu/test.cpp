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

// Silu reference implementation
std::bfloat16_t silu_bf16(std::bfloat16_t &input) {
  // Compute tanh approximation
  std::bfloat16_t half_x = input * std::bfloat16_t(0.5f);
  std::bfloat16_t tanh_half_x = std::tanh(half_x);
  std::bfloat16_t sigmoid_approx =
      std::bfloat16_t(0.5f) * (tanh_half_x + std::bfloat16_t(1.0f));

  // Compute output: x * tanh_approx
  return input * sigmoid_approx;
}

// SwiGLU reference implementation
std::bfloat16_t swiglu_bf16(std::bfloat16_t &input, std::bfloat16_t &w1,
                            std::bfloat16_t &w2) {
  // Compute the first part: x * w1
  std::bfloat16_t x_w1 = input * w1;
  // Compute the second part: x * w2
  std::bfloat16_t x_w2 = input * w2;
  // Apply the silu activation function to the second part
  std::bfloat16_t silu_output = silu_bf16(x_w2);
  // Compute the final output: x * w1 * silu_output
  return x_w1 * silu_output;
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

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

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

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, N * sizeof(std::bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_weights = xrt::bo(device, 2 * N * sizeof(std::bfloat16_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, N * sizeof(std::bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects." << std::endl;

  std::bfloat16_t *bufInA = bo_inA.map<std::bfloat16_t *>();
  std::vector<std::bfloat16_t> srcVecA;
  for (int i = 0; i < N; i++)
    srcVecA.push_back(std::bfloat16_t(i * 0.05f + -1.0f)); // Example data
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(std::bfloat16_t)));

  // Generate the W1 and W2 weights
  std::vector<std::bfloat16_t> srcVecW1;
  std::vector<std::bfloat16_t> srcVecW2;
  for (int i = 0; i < N; i++) {
    // Example weights, can be replaced with actual model weights
    srcVecW1.push_back(std::bfloat16_t(0.1f * (i % 10) + 0.1f));
    srcVecW2.push_back(std::bfloat16_t(0.2f * (i % 20) + 0.2f));
  }
  std::vector<std::bfloat16_t> srcVecWeights;
  // Interleave the weights into one vector in 1024 elements chunks
  // of each W1 and W2
  for (int i = 0; i < N; i += 1024) {
    for (int j = 0; j < 1024 && (i + j) < N; j++) {
      srcVecWeights.push_back(srcVecW1[i + j]);
    }
    for (int j = 0; j < 1024 && (i + j) < N; j++) {
      srcVecWeights.push_back(srcVecW2[i + j]);
    }
  }

  // Write the weights to the buffer object
  auto bufWeights = bo_weights.map<std::bfloat16_t *>();
  memcpy(bufWeights, srcVecWeights.data(),
         srcVecWeights.size() * sizeof(std::bfloat16_t));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  // Setup run to configure
  auto cfg_run =
      kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_weights, bo_out);
  cfg_run.wait();
  auto start = std::chrono::high_resolution_clock::now();
  // Test run
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_weights, bo_out);
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
    std::bfloat16_t ref = swiglu_bf16(srcVecA[i], srcVecW1[i], srcVecW2[i]);
    if (!test_utils::nearly_equal(*(bufOut + i), ref, 0.05f)) {
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
