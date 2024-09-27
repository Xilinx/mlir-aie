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

#define MATRIX_ROWS 7
#define MATRIX_COLS 19

#define SIZE (MATRIX_ROWS * MATRIX_COLS * sizeof(int32_t)) // in bytes

void print_matrix(int32_t *buf, int n_rows, int n_cols) {
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      std::cout << std::setw(4) << buf[row * n_cols + col] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v = test_utils::load_instr_sequence(INSTS_TXT);
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
  auto bo_in =
      xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int32_t *buf_in = bo_in.map<int32_t *>();
  for (int i = 0; i < SIZE / sizeof(buf_in[0]); i++) {
    buf_in[i] = i; // even
  }
  int32_t *buf_out = bo_out.map<int32_t *>();
  memset(buf_out, 0, SIZE);

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "Input:" << std::endl;
  print_matrix(buf_in, MATRIX_ROWS, MATRIX_COLS);

  int32_t ref[MATRIX_ROWS * MATRIX_COLS] = {};
  for (int row = 0; row < MATRIX_ROWS; row++) {
    for (int col = 0; col < MATRIX_COLS; col++) {
      ref[col * MATRIX_ROWS + row] = row * MATRIX_COLS + col;
    }
  }
  std::cout << "Expected:" << std::endl;
  print_matrix(ref, MATRIX_COLS, MATRIX_ROWS);
  std::cout << "Output:" << std::endl;
  print_matrix(buf_out, MATRIX_COLS, MATRIX_ROWS);

  if (memcmp(ref, buf_out, sizeof(ref)) == 0) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL." << std::endl;
  }

  return 0;
}
