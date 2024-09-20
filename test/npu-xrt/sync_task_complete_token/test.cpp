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

#include "test_utils.h"

#ifndef XCLBIN
#define XCLBIN "final.xclbin"
#endif

#ifndef INSTS_TXT
#define INSTS_TXT "insts.txt"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define DTYPE int32_t

#define OUTPUT_LEN (4 * 4)
#define INPUT_LEN (16 * OUTPUT_LEN)

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v = test_util::load_instr_sequence(INSTS_TXT);
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
  auto bo_input = xrt::bo(device, INPUT_LEN * sizeof(DTYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, OUTPUT_LEN * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < INPUT_LEN; i++) {
    buf_input[i] = i;
  }
  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, OUTPUT_LEN * sizeof(DTYPE));

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

  DTYPE ref[OUTPUT_LEN] = {};
  for (int i = 0; i < OUTPUT_LEN; i++) {
    for (int j = 0; j < 16; j++) {
      ref[i] += buf_input[i * 16 + j];
    }
  }
  std::cout << "Reference: ";
  for (int i = 0; i < OUTPUT_LEN; i++) {
    std::cout << std::setw(4) << (long)ref[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Output:    ";
  for (int i = 0; i < OUTPUT_LEN; i++) {
    std::cout << std::setw(4) << (long)buf_output[i] << " ";
  }
  std::cout << std::endl;

  if (memcmp(ref, buf_output, sizeof(ref)) == 0) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL." << std::endl;
  }

  return 0;
}
