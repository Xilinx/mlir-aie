// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test for one parameter n as both the offset and the length of a transfer.
//
// Setup:
//   - Input buffer: 1024 i32 values [0, 1, ..., 1023]
//   - The DMAs move 16 + 16 * n values starting at value n into the output
//   - Output buffer: 1024 i32 values, prefilled with -1
//
// We run with several values of n and check both the values moved and that
// nothing past them was written.
//

#include <cstdint>
#include <iostream>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

int main(int argc, const char *argv[]) {
  constexpr int N = 1024;
  constexpr int UNIT = 16;
  constexpr int STATIC_LEN = 16;

  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *buf_in = bo_in.map<int32_t *>();
  for (int i = 0; i < N; ++i)
    buf_in[i] = i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  bool all_pass = true;
  for (int32_t n : {0, 3, 1, 0, 30}) {
    for (int i = 0; i < N; ++i)
      buf_out[i] = -1;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    params.write("n", n);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int moved = STATIC_LEN + n * UNIT;
    int errors = 0;
    for (int i = 0; i < N; ++i) {
      int32_t expected = i < moved ? n + i : int32_t(-1);
      if (buf_out[i] != expected) {
        if (errors < 8)
          std::cout << "  out[" << i << "] = " << buf_out[i] << ", expected "
                    << expected << std::endl;
        ++errors;
      }
    }
    bool pass = errors == 0;
    if (!pass)
      all_pass = false;
    std::cout << "n=" << n << "  moved " << moved << " values  "
              << (pass ? "PASS" : "FAIL") << std::endl;
  }

  if (all_pass) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
