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

#ifndef INSTS_BIN
#define INSTS_BIN "build/insts.bin"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define IN_SIZE (M * N * sizeof(float))  // in bytes
#define BIAS_SIZE (N * sizeof(float))    // in bytes
#define OUT_SIZE (M * N * sizeof(float)) // in bytes

void print_matrix(float *buf, int n_rows, int n_cols) {
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      std::cout << std::setw(4) << buf[row * n_cols + col] << " ";
    }
    std::cout << std::endl;
  }
}

std::vector<uint32_t> load_instr_binary(std::string instr_path) {
  // Open file in binary mode
  std::ifstream instr_file(instr_path, std::ios::binary);
  if (!instr_file.is_open()) {
    throw std::runtime_error("Unable to open instruction file\n");
  }

  // Get the size of the file
  instr_file.seekg(0, std::ios::end);
  std::streamsize size = instr_file.tellg();
  instr_file.seekg(0, std::ios::beg);

  // Check that the file size is a multiple of 4 bytes (size of uint32_t)
  if (size % 4 != 0) {
    throw std::runtime_error("File size is not a multiple of 4 bytes\n");
  }

  // Allocate vector and read the binary data
  std::vector<uint32_t> instr_v(size / 4);
  if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), size)) {
    throw std::runtime_error("Failed to read instruction file\n");
  }
  return instr_v;
}

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v = load_instr_binary(INSTS_BIN);
  assert(instr_v.size() > 0);

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(std::string(XCLBIN));

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
      xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_bias =
      xrt::bo(device, BIAS_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  float *buf_in = bo_in.map<float *>();
  for (int i = 0; i < IN_SIZE / sizeof(buf_in[0]); i++) {
    buf_in[i] = i;
  }
  float *buf_bias = bo_bias.map<float *>();
  for (int i = 0; i < BIAS_SIZE / sizeof(buf_bias[0]); i++) {
    buf_bias[i] = 3 * i;
  }
  float *buf_out = bo_out.map<float *>();
  memset(buf_out, 0, OUT_SIZE);

  // Instruction buffer for DMA configuration
  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_bias, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  float ref[M * N] = {};
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ref[i * N + j] = buf_in[i * N + j] + buf_bias[j];
    }
  }

  if (M <= 64 && N <= 64) {
    std::cout << "Input:" << std::endl;
    print_matrix(buf_in, M, N);
    std::cout << "Bias:" << std::endl;
    print_matrix(buf_bias, 1, N);
    std::cout << "Expected:" << std::endl;
    print_matrix(ref, M, N);
    std::cout << "Output:" << std::endl;
    print_matrix(buf_out, M, N);
  }

  if (memcmp(ref, buf_out, sizeof(ref)) == 0) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL." << std::endl;
  }

  return 0;
}
