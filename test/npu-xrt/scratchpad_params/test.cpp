// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test host application for scratchpad register-write use case.
//
// Demonstrates passing an arbitrary runtime parameter to an AIE core via:
//   Host scratchpad → UPDATE_REG → core local buffer → ObjectFIFO → DDR output
//
// Synchronization uses a lock: the runtime sequence sets the lock after
// UPDATE_REG completes, and the core blocks on lock acquire until then.
//

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdfloat>
#include <string>
#include <unistd.h>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

constexpr std::bfloat16_t FOO_1 = (std::bfloat16_t)3.0;
constexpr std::bfloat16_t BAR_1 = (std::bfloat16_t)4.0;
constexpr std::bfloat16_t FOO_2 = (std::bfloat16_t)2.0;
constexpr std::bfloat16_t BAR_2 = (std::bfloat16_t)5.0;

int main(int argc, const char *argv[]) {
  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_out = xrt::ext::bo{device, 2 * sizeof(std::bfloat16_t)};
  auto *buf_out = bo_out.map<std::bfloat16_t *>();
  memset(buf_out, 0, 2 * sizeof(std::bfloat16_t));
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_out);
  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  // Run 1: 3.0 * 4.0 = 12.0
  params.write("foo", FOO_1);
  params.write("bar", BAR_1);
  params.sync();

  run.start();
  run.wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::bfloat16_t result1 = buf_out[0];
  std::bfloat16_t expected1 = FOO_1 * BAR_1;

  std::cout << "Run 1 — Expected: " << expected1 << ", Got: " << result1
            << std::endl;

  // Run 2: 2.0 * 5.0 = 10.0
  params.write("foo", FOO_2);
  params.write("bar", BAR_2);
  params.sync();
  memset(buf_out, 0, 2 * sizeof(std::bfloat16_t));
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  run.start();
  run.wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::bfloat16_t result2 = buf_out[0];
  std::bfloat16_t expected2 = FOO_2 * BAR_2;

  std::cout << "Run 2 — Expected: " << expected2 << ", Got: " << result2
            << std::endl;

  if (result1 == expected1 && result2 == expected2) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
