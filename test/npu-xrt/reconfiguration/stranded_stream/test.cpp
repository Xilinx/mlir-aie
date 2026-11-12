// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs aie.mlir's main:sequence three times and checks that @leave writes its
// first object + 100 and @use its input + 100.

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int IN = 64, OUT = 32, SLICE = 16, USE_IN = 48;

int main() {
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  xrt::bo bo_in = xrt::ext::bo{device, IN * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, OUT * sizeof(int32_t)};
  int32_t *in = bo_in.map<int32_t *>();
  int32_t *out = bo_out.map<int32_t *>();
  for (int i = 0; i < IN; i++)
    in[i] = 1000 * (i >= USE_IN) + i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int failures = 0;
  for (int pass = 0; pass < 3; pass++) {
    std::fill(out, out + OUT, -1);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto kernel = xrt::ext::kernel(context, "main:sequence");
    xrt::run run(kernel);
    run.set_arg(0, bo_in);
    run.set_arg(1, bo_out);
    run.start();
    run.wait2();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int wrong = 0;
    for (int m = 0; m < SLICE; m++)
      wrong +=
          (out[m] != in[m] + 100) + (out[SLICE + m] != in[USE_IN + m] + 100);
    if (wrong) {
      std::cout << "pass " << pass << ": " << wrong << " wrong; @use got";
      for (int m = 0; m < SLICE; m++)
        std::cout << " " << out[SLICE + m];
      std::cout << std::endl;
      failures++;
    }
  }
  std::cout << (failures ? "FAIL!" : "PASS!") << std::endl;
  return failures ? 1 : 0;
}
