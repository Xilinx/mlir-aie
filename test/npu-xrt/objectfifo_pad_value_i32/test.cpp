//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the objectfifo pad_value test: an aie.objectfifo with
// padDimensions + padValue = 42 pads a 13-element i32 transfer to 16 (2 before,
// 1 after) as a pure DMA passthrough. See aie.mlir.

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 13;  // real i32 payload
constexpr int OUT_SIZE = 16; // 2 (before) + 13 + 1 (after)
constexpr int PAD_BEFORE = 2;
constexpr uint32_t PAD_VALUE = 42;
constexpr uint32_t POISON = 0xdeadbeef;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_pad_value_i32");
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
  auto bo_in = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  uint32_t *bufIn = bo_in.map<uint32_t *>();
  for (int i = 0; i < IN_SIZE; i++)
    bufIn[i] = i + 1; // 1..13, none equal to PAD_VALUE

  // Poison-fill the output so an unwritten slot is distinguishable from a pad.
  uint32_t *bufOut = bo_out.map<uint32_t *>();
  for (int i = 0; i < OUT_SIZE; i++)
    bufOut[i] = POISON;

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int j = 0; j < OUT_SIZE; j++) {
    uint32_t want;
    if (j < PAD_BEFORE || j >= PAD_BEFORE + IN_SIZE)
      want = PAD_VALUE; // pad region must carry the requested pad value
    else
      want = static_cast<uint32_t>(j - PAD_BEFORE + 1); // real data 1..13
    if (bufOut[j] != want) {
      std::cout << "out[" << j << "] = " << bufOut[j] << " != " << want << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
