//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// E2E test: Shim MM2S BD reuse with >16 fire-and-forget tasks.
// 20 MM2S tasks on one shim tile, alternating buf_a and buf_b.
// Core tile passthrough. Verifies output matches interleaved source data.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {
  cxxopts::Options options("shim_dma_bd_reuse");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  int verbosity = vm["verbosity"].as<int>();

  constexpr int SLICE = 256;
  constexpr int NUM_TRANSFERS = 20;
  constexpr int SLICES_PER_SRC = 10; // 10 from buf_a, 10 from buf_b
  constexpr int SRC_SIZE = SLICES_PER_SRC * SLICE; // 2560
  constexpr int OUT_SIZE = NUM_TRANSFERS * SLICE;   // 5120

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, SRC_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, SRC_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // buf_a: 1, 2, 3, ... 2560
  // buf_b: 10001, 10002, ... 12560
  int32_t *a = bo_a.map<int32_t *>();
  int32_t *b = bo_b.map<int32_t *>();
  for (int i = 0; i < SRC_SIZE; i++) {
    a[i] = i + 1;
    b[i] = 10000 + i + 1;
  }

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running: 20 MM2S tasks (>16 BD limit), BD reuse via await."
              << std::endl;

  auto run = kernel(3, bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *out = bo_out.map<int32_t *>();

  // Transfer pattern: t0=a[0:256], t1=b[0:256], t2=a[256:512], t3=b[256:512]...
  // Even transfers from buf_a, odd from buf_b. Each pair shares an offset.
  int errors = 0;
  for (int t = 0; t < NUM_TRANSFERS; t++) {
    bool is_b = (t & 1);
    int src_slice = t / 2;
    for (int j = 0; j < SLICE; j++) {
      int32_t expected = is_b ? (10000 + src_slice * SLICE + j + 1)
                              : (src_slice * SLICE + j + 1);
      int32_t actual = out[t * SLICE + j];
      if (actual != expected) {
        errors++;
        if (errors <= 10 && verbosity >= 1)
          std::cout << "Transfer " << t << "[" << j << "]: expected "
                    << expected << ", got " << actual << std::endl;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS! (20 MM2S fire-and-forget, BD reuse, 2 src buffers)\n"
              << std::endl;
    return 0;
  } else {
    std::cout << "\n" << errors << " mismatches.\nfail.\n" << std::endl;
    return 1;
  }
}
