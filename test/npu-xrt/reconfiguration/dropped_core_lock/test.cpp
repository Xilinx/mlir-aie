// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs main:sequence three times and checks @a's and @b's output slices; a run
// that does not finish in time fails.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int SLICE = 16;

int main() {
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  xrt::bo bo_in = xrt::ext::bo{device, SLICE * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, 2 * SLICE * sizeof(int32_t)};
  int32_t *in = bo_in.map<int32_t *>();
  int32_t *out = bo_out.map<int32_t *>();
  for (int i = 0; i < SLICE; i++)
    in[i] = 1000 + i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int failures = 0;
  for (int pass = 0; pass < 3; pass++) {
    std::fill(out, out + 2 * SLICE, -1);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto kernel = xrt::ext::kernel(context, "main:sequence");
    xrt::run run(kernel);
    run.set_arg(0, bo_in);
    run.set_arg(1, bo_out);
    run.start();
    if (run.wait(std::chrono::milliseconds(5000)) != ERT_CMD_STATE_COMPLETED) {
      std::cout << "pass " << pass << ": did not finish" << std::endl;
      std::cout << "FAIL!" << std::endl;
      return 1;
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int wrong = 0;
    for (int k = 0; k < SLICE; k++)
      wrong += (out[k] != k + 7) + (out[SLICE + k] != in[k] + 100);
    if (wrong) {
      std::cout << "pass " << pass << ": " << wrong << " wrong" << std::endl;
      failures++;
    }
  }
  std::cout << (failures ? "FAIL!" : "PASS!") << std::endl;
  return failures ? 1 : 0;
}
