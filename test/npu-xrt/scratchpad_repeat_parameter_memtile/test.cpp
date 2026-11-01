// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host side of the mem tile repeat_parameter test; aie.mlir describes the
// phases. Each call's output is checked value by value, including that nothing
// past each phase's values was written.

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

int main(int argc, const char *argv[]) {
  constexpr int IN_LEN = 2048;
  constexpr int OUT_LEN = 4096;
  constexpr int TILE = 64;

  auto device = xrt::device(0);

  std::string kernelName = "test:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  xrt::bo bo_in = xrt::ext::bo{device, IN_LEN * sizeof(int32_t)};
  auto *buf_in = bo_in.map<int32_t *>();
  for (int i = 0; i < IN_LEN; ++i)
    buf_in[i] = i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::bo bo_out = xrt::ext::bo{device, OUT_LEN * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  bool all_pass = true;
  for (auto [lng, shrt] :
       {std::pair{15, 0}, {0, 15}, {6, 2}, {2, 6}, {15, 15}, {0, 0}, {9, 1}}) {
    for (int i = 0; i < OUT_LEN; ++i)
      buf_out[i] = -1;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    params.write("long", lng);
    params.write("long_rows", lng + 1);
    params.write("short", shrt);
    params.write("short_rows", shrt + 1);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (int i = 0; i < OUT_LEN; ++i) {
      int phase = i / 1024, j = i % 1024;
      int filled = phase == 0   ? 2 * TILE
                   : phase == 1 ? (lng + 1) * TILE
                   : phase == 2 ? (shrt + 1) * TILE
                                : 0;
      int src = phase == 2 ? 1024 + j / 64 * 64 + j % 8 * 8 + j % 64 / 8 : j;
      int32_t expected = j < filled ? src : -1;
      if (buf_out[i] != expected) {
        if (errors < 8)
          std::cout << "  out[" << i << "] = " << buf_out[i] << ", expected "
                    << expected << std::endl;
        ++errors;
      }
    }
    bool pass = errors == 0;
    all_pass &= pass;
    std::cout << "long=" << lng << " short=" << shrt << "  "
              << (pass ? "PASS" : "FAIL") << std::endl;
  }

  std::cout << (all_pass ? "PASS!" : "FAIL.") << std::endl;
  return all_pass ? 0 : 1;
}
