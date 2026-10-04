// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int RUNS = 100;
constexpr size_t COUNT = 512;

int main() {
  auto device = xrt::device(0);
  xrt::hw_context context(device, xrt::elf{"aie.elf"});
  auto kernel = xrt::ext::kernel(context, "add_two:sequence");
  xrt::bo bo = xrt::ext::bo{device, COUNT * sizeof(int32_t)};
  auto *buf = bo.map<int32_t *>();
  auto run = xrt::run(kernel);
  run.set_arg(0, bo);

  std::vector<double> us;
  for (int r = 0; r < RUNS; r++) {
    for (size_t i = 0; i < COUNT; i++)
      buf[i] = int32_t(r * COUNT + i);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto stop = std::chrono::high_resolution_clock::now();
    us.push_back(
        std::chrono::duration<double, std::micro>(stop - start).count());
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (size_t i = 0; i < COUNT; i++) {
      if (buf[i] != int32_t(r * COUNT + i + 2)) {
        std::cout << "run " << r << ", element " << i << ": " << buf[i]
                  << " != " << r * COUNT + i + 2 << "\nFail." << std::endl;
        return 1;
      }
    }
  }
  std::vector<double> later(us.begin() + 1, us.end());
  std::nth_element(later.begin(), later.begin() + later.size() / 2,
                   later.end());
  std::cout << "first run: " << us[0]
            << " us, median later run: " << later[later.size() / 2]
            << " us\nPASS!" << std::endl;
  return 0;
}
