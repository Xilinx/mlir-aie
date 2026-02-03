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
      "path of file containing userspace instructions to be sent to the "
      "command processor",
      cxxopts::value<std::string>())(
      "length,l", "the length of the transfer in bytes",
      cxxopts::value<int>()->default_value("1024"))(
      "trace,t", "enable tracing (0 or 1)",
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

  int enable_trace = vm["trace"].as<int>();

  int N = vm["length"].as<int>();
  if ((N % 4)) {
    std::cerr << "Length must be a multiple of 4 bytes." << std::endl;
    return 1;
  }

  int N_int32 = N / 4; // Convert bytes to int32 elements (1KB = 256 int32)
  int N_read = N * 4;  // Read buffer is 4KB

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
  // Placeholder buffers
  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp2 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  // Trace buffer (8KB if enabled, 1 byte otherwise)
  constexpr int trace_size = 16384;
  int actual_trace_size = enable_trace ? trace_size : 1;
  auto bo_trace = xrt::bo(device, actual_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

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
  for (int i = 0; i < N_read / 4; i++) {
    bufB[i] = i;
  }
  // Initialize trace buffer if enabled
  if (enable_trace) {
    char *bufTrace = bo_trace.map<char *>();
    memset(bufTrace, 0, trace_size);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel." << std::endl;
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_A, bo_B, bo_tmp1, bo_tmp2, bo_trace);
  run.wait();

  bo_A.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (enable_trace) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  bufA = bo_A.map<uint32_t *>();

  int errors = 0;

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
  
  // Write trace data to file if enabled
  if (enable_trace) {
    char *bufTrace = bo_trace.map<char *>();
    std::ofstream trace_file("trace.txt");
    uint32_t *trace_data = reinterpret_cast<uint32_t *>(bufTrace);
    for (int i = 0; i < trace_size / 4; i++) {
      if (trace_data[i] != 0) {
        trace_file << std::hex << trace_data[i] << std::endl;
      }
    }
    trace_file.close();
    
    if (verbosity >= 1)
      std::cout << "Trace data written to trace.txt" << std::endl;
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
