// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

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

#define INPUT_SIZE (100 * sizeof(int))  // in bytes
#define OUTPUT_SIZE (100 * sizeof(int)) // in bytes
#define WIDTH_SIZE (10 * sizeof(int))   // in bytes

#define INPUT_ROWS INPUT_SIZE / WIDTH_SIZE
#define OUTPUT_ROWS OUTPUT_SIZE / WIDTH_SIZE

#include "test_utils.h"

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Sliding Window Conditional Test");
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
      "length,l", "the length of the transfer in int32_t",
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

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

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
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << std::endl;
  auto kernel = xrt::kernel(context, kernelName);
  
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int *buf_input = bo_input.map<int *>();
  std::cout << std::endl << std::endl << "Input: " << std::endl;
  for (int i = 0; i < INPUT_ROWS; i++) {
    std::cout << "row " << i << " : ";
    for (int j = 0; j < WIDTH_SIZE / sizeof(buf_input[0]); j++) {
      buf_input[i * INPUT_ROWS + j] = i;
      std::cout << buf_input[i * INPUT_ROWS + j] << " ";
    }
    std::cout << std::endl << std::endl;
  }
  int *buf_output = bo_output.map<int *>();
  memset(buf_output, 0, OUTPUT_SIZE);

  // Instruction buffer for DMA configuration
  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  bool pass = true;
  std::cout << std::endl << "Output: " << std::endl;
  for (int i = 0; i < OUTPUT_ROWS; i++) {
    std::cout << "row " << i << std::endl;
    for (int j = 0; j < WIDTH_SIZE / sizeof(buf_output[0]); j++) {
      int expected_output = 0;
      if (i == 0) {
        expected_output = buf_input[i * INPUT_ROWS] * 2;
      } else {
        expected_output =
            buf_input[(i - 1) * INPUT_ROWS] + buf_input[i * INPUT_ROWS];
      }
      std::cout << "expected: " << expected_output << ", ";
      std::cout << "got: " << buf_output[i * OUTPUT_ROWS + j] << std::endl;
      pass &= buf_output[i * OUTPUT_ROWS + j] == expected_output;
    }
    std::cout << std::endl << std::endl;
  }
  std::cout << std::endl << std::endl;
  std::cout << (pass ? "PASS!" : "FAIL.") << std::endl;

  return !pass;
}