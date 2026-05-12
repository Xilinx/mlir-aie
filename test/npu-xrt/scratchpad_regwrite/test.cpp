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

#include <cstdint>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define DTYPE int32_t

// The runtime parameter we want to pass to the core.
// Must be in range [0, 2^30 - 1].
constexpr uint32_t TEST_VALUE_1 = 42;
constexpr uint32_t TEST_VALUE_2 = 43;

int main(int argc, const char *argv[]) {
  auto device = xrt::device(0);

  std::string kernelName = "regwrite_test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  // Output buffer (1 x i32)
  xrt::bo bo_out = xrt::ext::bo{device, sizeof(DTYPE)};
  auto *buf_out = bo_out.map<DTYPE *>();
  buf_out[0] = 0;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Create run object and set arguments
  auto run = xrt::run(kernel);
  run.set_arg(0, bo_out); // %out

  // Write the test value (shifted left by 2) into scratchpad StateTable[0].
  // The firmware will mask the lower 2 bits after writing, so pre-shifting
  // ensures no data is lost. The core right-shifts by 2 to recover.
  auto scratchpad_bo = run.get_ctrl_scratchpad_bo();
  auto *s_map = scratchpad_bo.map<uint32_t *>();
  s_map[0] = TEST_VALUE_1 << 2;
  scratchpad_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Writing to scratchpad: " << TEST_VALUE_1
            << " (shifted: " << (TEST_VALUE_1 << 2) << ")" << std::endl;

  // Run
  auto t_start = std::chrono::high_resolution_clock::now();
  run.start();
  run.wait2();
  auto t_stop = std::chrono::high_resolution_clock::now();

  // Read back output
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  DTYPE result = buf_out[0];

  float time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time: " << std::fixed << std::setprecision(0)
            << std::setw(8) << time_us << " us" << std::endl;
  std::cout << "Expected: " << TEST_VALUE_1 << ", Got: " << result << std::endl;

  // Run 2
  // read back current scratchpad value
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  scratchpad_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t readback = s_map[0];
  std::cout << "Current scratchpad value (raw): " << readback
            << " (shifted: " << (readback >> 2) << ")" << std::endl;

  // UPDATE_REG is additive: it adds the scratchpad delta to the existing
  // register value. To change from old_value to new_value, we must write
  // (new - old) << 2 into the scratchpad, not new << 2.
  uint32_t delta = TEST_VALUE_2 - TEST_VALUE_1;
  std::cout << "Writing to scratchpad: " << TEST_VALUE_2
            << " (shifted delta/raw: " << (delta << 2) << ")" << std::endl;
  s_map[0] = delta << 2;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  scratchpad_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  t_start = std::chrono::high_resolution_clock::now();
  run.start();
  run.wait2();
  t_stop = std::chrono::high_resolution_clock::now();

  // Read back output
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  result = buf_out[0];

  time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  std::cout << "Elapsed time: " << std::fixed << std::setprecision(0)
            << std::setw(8) << time_us << " us" << std::endl;
  std::cout << "Expected: " << TEST_VALUE_2 << ", Got: " << result << std::endl;

  if (result == static_cast<DTYPE>(TEST_VALUE_2)) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
