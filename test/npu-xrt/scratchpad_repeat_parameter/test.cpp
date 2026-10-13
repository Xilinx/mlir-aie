// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Host side of the repeat_parameter test; aie.mlir describes the transfers.
// Each repeat count is checked value by value, including that nothing past
// the moved values was written, and out-of-range counts must be rejected.

#include <cstdint>
#include <iostream>
#include <stdexcept>
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
  constexpr int TILE = 64;
  constexpr int MAX_REPS = 15;

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

  xrt::bo bo_out = xrt::ext::bo{device, 2 * N * sizeof(int32_t)};
  auto *buf_out = bo_out.map<int32_t *>();
  xrt::bo bo_out2 = xrt::ext::bo{device, 2 * N * sizeof(int32_t)};
  auto *buf_out2 = bo_out2.map<int32_t *>();

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_in);
  run.set_arg(1, bo_out);
  run.set_arg(2, bo_out2);

  auto params = test_utils::ParameterScratchpad(run, "params.txt");

  bool all_pass = true;
  for (int32_t reps : {2, 0, 6, MAX_REPS, 2}) {
    for (int i = 0; i < 2 * N; ++i) {
      buf_out[i] = -1;
      buf_out2[i] = -1;
    }
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    params.write("reps", reps);
    params.write("rows", reps + 1);
    params.sync();

    run.start();
    run.wait2();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int moved = (reps + 1) * TILE;
    int errors = 0;
    // The plain pushes' two tiles land in the second half.
    for (int i = 0; i < 2 * N; ++i) {
      int j = i < N ? i : i - N;
      int filled = i < N ? moved : 2 * TILE;
      int32_t expected = j < filled ? j % TILE : int32_t(-1);
      if (buf_out[i] != expected) {
        if (errors < 8)
          std::cout << "  out[" << i << "] = " << buf_out[i] << ", expected "
                    << expected << std::endl;
        ++errors;
      }
      int32_t expected2 = j < filled ? j : int32_t(-1);
      if (buf_out2[i] != expected2) {
        if (errors < 8)
          std::cout << "  out2[" << i << "] = " << buf_out2[i] << ", expected "
                    << expected2 << std::endl;
        ++errors;
      }
    }
    bool pass = errors == 0;
    if (!pass)
      all_pass = false;
    std::cout << "reps=" << reps << "  moved " << moved << " values  "
              << (pass ? "PASS" : "FAIL") << std::endl;
  }

  // The tasks' repeat count, 15, bounds the parameter, and the host library
  // rejects anything else before it is written.
  for (int32_t reps : {MAX_REPS + 1, -1}) {
    bool rejected = false;
    try {
      params.write("reps", reps);
    } catch (const std::invalid_argument &e) {
      rejected = true;
      std::cout << "reps=" << reps << "  rejected: " << e.what() << std::endl;
    }
    if (!rejected || params.read("reps") != uint32_t(2) << 16) {
      std::cout << "reps=" << reps << "  not rejected  FAIL" << std::endl;
      all_pass = false;
    }
  }

  if (all_pass) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }
}
