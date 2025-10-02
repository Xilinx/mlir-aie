// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// This file is adapted from
// programming_examples/basic/shuffle_transpose/test.cpp; see that file for
// commit history and original authors.

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cxxopts.hpp"
#include "test_utils.h"

#if !defined(DTYPE_i8) && !defined(DTYPE_i16) && !defined(DTYPE_i32)
#error Please specify data type at kernel compile time using e.g., -DDTYPE_i8 or -DDTYPE_i16 or -DDTYPE_i32.
#endif

#if defined(DTYPE_i8)
#define DTYPE uint8_t
#endif
#if defined(DTYPE_i16)
#define DTYPE uint16_t
#endif
#if defined(DTYPE_i32)
#define DTYPE uint32_t
#endif

/* This example performs a 16x16 INT8 transpose.
   M and N are passed in as 16 in Makefile run cmd.
   kernel.cc includes an AIE kernel that is specific to 16x16 */
template <typename T>
void print_matrix(T *buf, int n_rows, int n_cols) {
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      std::cout << std::setw(4) << (int)buf[row * n_cols + col] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  cxxopts::Options options("Shuffle Transpose Test",
                           "Test the Shuffle transpose kernel");

  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>())(
      "rows,M", "M, number of rows in the input matrix",
      cxxopts::value<int>()->default_value("64"))(
      "cols,N", "N, number of columns in the input matrix",
      cxxopts::value<int>()->default_value("64"));

  auto vm = options.parse(argc, argv);

  // Check required options
  if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
    std::cerr << "Error: Required options missing\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  assert(instr_v.size() > 0);

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  // Get the kernel from the xclbin
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });

  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), Node.c_str()) == 0);

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  uint32_t M = vm["M"].as<int>();
  uint32_t N = vm["N"].as<int>();

  size_t total_size = M * N * sizeof(DTYPE);

  auto bo_in =
      xrt::bo(device, total_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, total_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_in = bo_in.map<DTYPE *>();
  for (uint32_t i = 0; i < total_size / sizeof(buf_in[0]); i++) {
    buf_in[i] = (DTYPE)i;
  }

  DTYPE *buf_out = bo_out.map<DTYPE *>();
  memset(buf_out, 0, total_size);

  // Instruction buffer for DMA configuration
  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto t_start = std::chrono::system_clock::now();
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  auto t_stop = std::chrono::system_clock::now();
  float t_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  DTYPE *ref = (DTYPE *)calloc(M * N, sizeof(DTYPE));
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ref[j * M + i] = buf_in[i * N + j];
    }
  }

  if (M * N <= 4096) {
    std::cout << "Input:" << std::endl;
    print_matrix(buf_in, M, N);
    std::cout << "Expected:" << std::endl;
    print_matrix(ref, N, M);
    std::cout << "Output:" << std::endl;
    print_matrix(buf_out, N, M);
  }

  float throughput = (total_size / t_elapsed / 1e3); // GB/s
  std::cout << "Elapsed: " << t_elapsed << " us ";
  std::cout << "(" << throughput << " GB/s)" << std::endl;

  if (memcmp(ref, buf_out, total_size) == 0) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }

  return 0;
}
