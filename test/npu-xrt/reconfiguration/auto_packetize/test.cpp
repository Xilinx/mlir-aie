// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs aie.mlir's sequence three times and checks that configuration 1 wrote
// a + b and configuration 2 wrote a - b to their slices of c.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int SLICE = 16;
constexpr int COUNT = 2 * SLICE;
constexpr int32_t POISON = -1;

int main() {
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  xrt::bo bo_a = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  xrt::bo bo_b = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  xrt::bo bo_c = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  int32_t *a = bo_a.map<int32_t *>();
  int32_t *b = bo_b.map<int32_t *>();
  int32_t *c = bo_c.map<int32_t *>();
  for (int i = 0; i < COUNT; i++) {
    a[i] = 1000 + 3 * i;
    b[i] = 7 * i + 1;
  }
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto kernel = xrt::ext::kernel(context, "main:sequence");
  int failures = 0;
  for (int pass = 0; pass < 3; pass++) {
    std::fill(c, c + COUNT, POISON);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run(kernel);
    run.set_arg(0, bo_a);
    run.set_arg(1, bo_b);
    run.set_arg(2, bo_c);
    run.start();
    run.wait2();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int wrong[2] = {0, 0};
    for (int i = 0; i < COUNT; i++) {
      int32_t expected = i < SLICE ? a[i] + b[i] : a[i] - b[i];
      wrong[i / SLICE] += c[i] != expected;
    }
    for (int config = 0; config < 2; config++)
      if (wrong[config]) {
        std::cout << "pass " << pass << ", configuration " << config + 1 << ": "
                  << wrong[config] << " wrong" << std::endl;
        failures++;
      }
  }

  std::cout << (failures ? "FAIL!" : "PASS!") << std::endl;
  return failures ? 1 : 0;
}
