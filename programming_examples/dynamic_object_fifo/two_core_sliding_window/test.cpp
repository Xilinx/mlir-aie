// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef XCLBIN
#define XCLBIN "build/final.xclbin"
#endif

#ifndef INSTS_TXT
#define INSTS_TXT "build/insts.txt"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define INPUT_SIZE  (100 * sizeof(int)) // in bytes
#define OUTPUT_SIZE (100 * sizeof(int)) // in bytes
#define WIDTH_SIZE  (10 * sizeof(int))  // in bytes
#define INPUT_ROWS  INPUT_SIZE / WIDTH_SIZE
#define OUTPUT_ROWS OUTPUT_SIZE / WIDTH_SIZE

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v = load_instr_sequence(INSTS_TXT);
  assert(instr_v.size() > 0);

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);

  // Get the kernel from the xclbin
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), KERNEL_NAME) == 0);

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int *buf_input = bo_input.map<int *>();
  std::cout << std::endl << std::endl << "Input: " << std::endl;
  for(int i = 0; i < INPUT_ROWS; i++) {
    std::cout << "row " << i << " : ";
    for(int j = 0; j < WIDTH_SIZE / sizeof(buf_input[0]); j++) {
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
  for(int i = 0; i < OUTPUT_ROWS; i++) {
    std::cout << "row " << i << std::endl;
    for(int j = 0; j < WIDTH_SIZE / sizeof(buf_output[0]); j++) {
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

  return 0;
}