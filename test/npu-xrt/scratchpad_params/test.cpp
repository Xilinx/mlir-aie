// (c) Copyright 2026 Advanced Micro Devices, Inc.
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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

constexpr std::uint16_t FOO_1 = 0x4040; // bf16 3.0
constexpr std::uint16_t BAR_1 = 0x4080; // bf16 4.0
constexpr std::uint16_t FOO_2 = 0x4000; // bf16 2.0
constexpr std::uint16_t BAR_2 = 0x40a0; // bf16 5.0
constexpr std::uint16_t EXPECTED_1 = 0x4140; // bf16 12.0
constexpr std::uint16_t EXPECTED_2 = 0x4120; // bf16 10.0

int main(int argc, const char *argv[]) {
  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_out = xrt::ext::bo{device, 2 * sizeof(std::uint16_t)};
  auto *buf_out = bo_out.map<std::uint16_t *>();
  memset(buf_out, 0, 2 * sizeof(std::uint16_t));
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
  std::uint16_t result1 = buf_out[0];

  std::cout << "Run 1 - Expected bits: 0x" << std::hex << EXPECTED_1
            << ", Got: 0x" << result1 << std::dec << std::endl;

  // Run 2: 2.0 * 5.0 = 10.0
  params.write("foo", FOO_2);
  params.write("bar", BAR_2);
  params.sync();
  memset(buf_out, 0, 2 * sizeof(std::uint16_t));
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  run.start();
  run.wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::uint16_t result2 = buf_out[0];

  std::cout << "Run 2 - Expected bits: 0x" << std::hex << EXPECTED_2
            << ", Got: 0x" << result2 << std::dec << std::endl;

  if (result1 == EXPECTED_1 && result2 == EXPECTED_2) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
