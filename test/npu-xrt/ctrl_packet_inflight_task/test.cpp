// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr size_t DATA_COUNT = 64;
constexpr size_t BUF_SIZE = DATA_COUNT * sizeof(int32_t);

int main(int argc, const char *argv[]) {
  std::vector<int32_t> vec_in(DATA_COUNT);
  for (size_t i = 0; i < DATA_COUNT; i++)
    vec_in[i] = static_cast<int32_t>(7 * i - 100);

  auto device = xrt::device(0);
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, "main:sequence");
  xrt::bo bo_in = xrt::ext::bo{device, BUF_SIZE};
  xrt::bo bo_out = xrt::ext::bo{device, BUF_SIZE};

  memcpy(bo_in.map<char *>(), vec_in.data(), BUF_SIZE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bo_out.map<char *>(), 0, BUF_SIZE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);
  run.start();
  run.wait2();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::vector<int32_t> vec_out(DATA_COUNT);
  memcpy(vec_out.data(), bo_out.map<char *>(), BUF_SIZE);
  int errors = 0;
  for (size_t i = 0; i < DATA_COUNT; i++) {
    int32_t ref = vec_in[i] + 1;
    if (vec_out[i] != ref) {
      if (errors < 16)
        std::cout << "index " << std::setw(3) << i << ": out " << vec_out[i]
                  << ", ref " << ref << std::endl;
      errors++;
    }
  }
  if (errors) {
    std::cout << errors << " mismatches.\nFail." << std::endl;
    return 1;
  }
  std::cout << "PASS!" << std::endl;
  return 0;
}
