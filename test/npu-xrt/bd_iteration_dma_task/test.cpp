//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Must match the #aie.bd_iteration attribute in aie.mlir.
constexpr int SIZE = 4;     // iteration_size (slots)
constexpr int START = 2;    // iteration_current
constexpr int CHUNK = 1024; // elements per execution
constexpr int N = SIZE * CHUNK;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("bd_iteration_dma_task");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernelName = vm["kernel"].as<std::string>();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  uint32_t *bufIn = bo_in.map<uint32_t *>();
  for (int i = 0; i < N; i++)
    bufIn[i] = i + 1;

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();
  int errors = 0;
  for (int k = 0; k < SIZE; k++) {
    int slot = (START + k) % SIZE;
    for (int j = 0; j < CHUNK; j++) {
      uint32_t want = bufIn[k * CHUNK + j];
      uint32_t got = bufOut[slot * CHUNK + j];
      if (got != want) {
        if (errors < 16)
          std::cout << "out[" << slot * CHUNK + j << "] = " << got
                    << " != " << want << "\n";
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\n" << errors << " mismatches.\n\nfailed.\n\n";
  return 1;
}
