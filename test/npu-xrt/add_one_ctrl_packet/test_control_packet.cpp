// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main() {
  constexpr int count = 8;
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo readback = xrt::ext::bo{device, count * sizeof(int32_t)};
  xrt::bo output = xrt::ext::bo{device, count * sizeof(int32_t)};
  auto *controlData = readback.map<int32_t *>();
  auto *dmaData = output.map<int32_t *>();
  int errors = 0;
  for (int pass = 0; pass < 3; ++pass) {
    std::fill_n(controlData, count, -1);
    std::fill_n(dmaData, count, -1);
    readback.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run(kernel);
    run.set_arg(0, readback);
    run.set_arg(1, output);
    run.start();
    run.wait2();
    readback.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < count; ++i) {
      if (dmaData[i] != 7 + i || controlData[i] != 8 + i) {
        std::cerr << "pass " << pass << ", element " << i << ": DMA "
                  << dmaData[i] << " (expected " << 7 + i << "), control "
                  << controlData[i] << " (expected " << 8 + i << ")\n";
        ++errors;
      }
    }
  }
  std::cout << (errors ? "FAIL!" : "PASS!") << std::endl;
  return errors ? 1 : 0;
}
