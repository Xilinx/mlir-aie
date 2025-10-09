// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define DTYPE int32_t

constexpr size_t DATA_COUNT = 256;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(DTYPE);

int main(int argc, const char *argv[]) {
  // Set up input data
  srand(1726250518);
  std::vector<DTYPE> vec_in(DATA_COUNT);
  for (int i = 0; i < vec_in.size(); i++) {
    vec_in[i] = DTYPE(rand());
  }

  // Set up XRT
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  // The name format here is <kernel_name>:<instance_name> from the config.json
  std::string kernelName = "add_two:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);
  xrt::bo bo_inout = xrt::ext::bo{device, BUF_SIZE};

  // Set up kernel run
  char *buf_inout = bo_inout.map<char *>();
  memcpy(buf_inout, vec_in.data(), BUF_SIZE);
  bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run = xrt::run(kernel);
  run.set_arg(0, bo_inout);

  // Run
  auto t_start = std::chrono::high_resolution_clock::now();
  run.start();
  run.wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();
  bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Validate output
  std::vector<DTYPE> vec_out(DATA_COUNT);
  std::vector<DTYPE> vec_ref(DATA_COUNT);
  memcpy(vec_out.data(), buf_inout, BUF_SIZE);
  for (int i = 0; i < DATA_COUNT; i++) {
    vec_ref[i] = vec_in[i] + 2;
  }
  bool outputs_correct = (vec_out == vec_ref);

  // Report results
  float time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time for all kernel executions: " << std::fixed
            << std::setprecision(0) << std::setw(8) << time << " μs"
            << std::endl;
  if (outputs_correct) {
    std::cout << "PASS!" << std::endl;
  } else {
    for (int i = 0; i < DATA_COUNT; i++) {
      std::cout << "in: " << std::setw(12) << vec_in[i] << ", "
                << "out: " << std::setw(12) << vec_out[i]
                << ", ref: " << std::setw(12) << vec_ref[i] << std::endl;
    }
    std::cout << "Fail." << std::endl;
  }

  return (outputs_correct ? 0 : 1);
}
