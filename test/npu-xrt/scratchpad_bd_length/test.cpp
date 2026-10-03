//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dispatches the length_parameter design several times on one run with
// different lengths. Each dispatch must move exactly 16 + extra words and
// leave the rest of the output untouched, including a return to a shorter
// length after a longer one.

#include <cstdint>
#include <iostream>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <parameter_scratchpad.h>

constexpr int N = 64;
constexpr int BASE = 16;
constexpr int32_t UNTOUCHED = -1;

int main() {
  auto device = xrt::device(0);
  xrt::elf elf{"aie.elf"};
  xrt::hw_context context(device, elf);
  auto kernel = xrt::ext::kernel(context, "test:sequence");

  xrt::bo bo_in = xrt::ext::bo{device, N * sizeof(int32_t)};
  xrt::bo bo_out = xrt::ext::bo{device, N * sizeof(int32_t)};
  auto *in = bo_in.map<int32_t *>();
  auto *out = bo_out.map<int32_t *>();
  for (int i = 0; i < N; i++)
    in[i] = 1000 + i;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);
  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  int errors = 0;
  for (uint32_t extra : {0u, 16u, 48u, 4u, 32u, 0u}) {
    for (int i = 0; i < N; i++)
      out[i] = UNTOUCHED;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    params.write("extra", extra);
    params.sync();
    run.start();
    run.wait2();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int len = BASE + static_cast<int>(extra);
    for (int i = 0; i < N; i++) {
      int32_t want = i < len ? 1000 + i : UNTOUCHED;
      if (out[i] != want) {
        std::cout << "extra " << extra << ": out[" << i << "] = " << out[i]
                  << ", expected " << want << "\n";
        errors++;
        break;
      }
    }
  }

  std::cout << (errors ? "FAIL!" : "PASS!") << std::endl;
  return errors ? 1 : 0;
}
