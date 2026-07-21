//===- test_runlist_hrx.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs two dispatches of the `x + 1` kernel as a single batched submit
// (hrx_test::dispatch_chain -> one ERT_CMD_CHAIN), where the second run
// consumes the first run's output:
//
//   run0:  out0 = inA  + 1     (inA[i] = i + 1  ->  out0[i] = i + 2)
//   run1:  out1 = out0 + 1     (out1[i] = i + 3)
//
// Validating out1 == i + 3 proves the in-chain producer->consumer dependency
// holds (run1 observed run0's on-device write). HRX-only: build with
// RUNTIME=hrx (the Makefile target forces -DUSE_HRX=ON).
//
//===----------------------------------------------------------------------===//

// Pulls in hrx_test_wrapper.h when TEST_UTILS_USE_HRX is defined.
#include "xrt_test_wrapper.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef TEST_UTILS_USE_HRX
#error "test_runlist_hrx.cpp requires the HRX backend (build with RUNTIME=hrx)."
#endif

int main(int argc, const char *argv[]) {
  using namespace hrx_test;
  constexpr int SIZE = 1024;

  struct args myargs = parse_args(argc, argv);

  try {
    Context &ctx = Context::get();
    LoadedKernel lk = load_kernel(myargs.xclbin, myargs.instr, myargs.kernel);

    Buffer bo_inA(ctx.stream, (size_t)SIZE * sizeof(int32_t));
    Buffer bo_out0(ctx.stream, (size_t)SIZE * sizeof(int32_t));
    Buffer bo_out1(ctx.stream, (size_t)SIZE * sizeof(int32_t));

    auto *pa = reinterpret_cast<int32_t *>(bo_inA.host_ptr());
    for (int i = 0; i < SIZE; ++i)
      pa[i] = i + 1;
    std::memset(bo_out0.host_ptr(), 0, (size_t)SIZE * sizeof(int32_t));
    std::memset(bo_out1.host_ptr(), 0, (size_t)SIZE * sizeof(int32_t));
    bo_inA.flush();
    bo_out0.flush();
    bo_out1.flush();

    // Two chained runs in one batched submit (single ERT_CMD_CHAIN); run1's
    // input is run0's output buffer.
    std::vector<ChainRun> chain = {
        {&lk, {&bo_inA, &bo_out0}},
        {&lk, {&bo_out0, &bo_out1}},
    };
    double us = dispatch_chain(chain);
    bo_out0.invalidate();
    bo_out1.invalidate();

    auto *p0 = reinterpret_cast<int32_t *>(bo_out0.host_ptr());
    auto *p1 = reinterpret_cast<int32_t *>(bo_out1.host_ptr());

    int errors = 0;
    std::cout << "Checking run 0 (out0 == in + 1)\n";
    for (int i = 0; i < SIZE; ++i) {
      int32_t ref = i + 2;
      if (p0[i] != ref) {
        if (errors < 10)
          std::cout << "  Error out0[" << i << "] " << p0[i] << " != " << ref
                    << "\n";
        errors++;
      }
    }
    std::cout << "Checking run 1 (out1 == out0 + 1)\n";
    for (int i = 0; i < SIZE; ++i) {
      int32_t ref = i + 3;
      if (p1[i] != ref) {
        if (errors < 10)
          std::cout << "  Error out1[" << i << "] " << p1[i] << " != " << ref
                    << "\n";
        errors++;
      }
    }

    std::cout << "\nChain NPU time: " << us << "us.\n";

    if (!errors) {
      std::cout << "\nPASS!\n\n";
      return 0;
    }
    std::cout << "\nError count: " << errors << "\n\nFailed.\n\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "\nHRX error: " << e.what() << "\n\nFailed.\n\n";
    return 1;
  }
}
