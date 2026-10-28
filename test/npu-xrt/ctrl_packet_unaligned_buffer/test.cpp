// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs aie.mlir's sequence twice and checks that every word of the buffer
// reaches the host.

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int COUNT = 256;
constexpr int32_t POISON = -1;

int main() {
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  xrt::bo bo_out = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  int32_t *out = bo_out.map<int32_t *>();

  auto kernel = xrt::ext::kernel(context, "main:sequence");
  int failures = 0;
  for (int pass = 0; pass < 2; pass++) {
    std::fill(out, out + COUNT, POISON);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run(kernel);
    run.set_arg(0, bo_out);
    run.start();
    run.wait2();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int wrong = 0;
    for (int i = 0; i < COUNT; i++)
      wrong += out[i] != i;
    if (wrong) {
      std::cout << "pass " << pass << ": " << wrong << " wrong" << std::endl;
      failures++;
    }
  }

  std::cout << (failures ? "FAIL!" : "PASS!") << std::endl;
  return failures ? 1 : 0;
}
