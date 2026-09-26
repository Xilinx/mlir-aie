//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the aiex.buffer_clear device test: one runtime sequence reads a
// core-tile accumulator into arg0[0:8] before the clear and arg0[8:16] after.
// See README.md.

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int BATCH = 8;
constexpr int OUT_SIZE = 2 * BATCH; // batch1 || batch2

int main(int argc, const char *argv[]) {
  cxxopts::Options options("buffer_clear_op");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Get a device handle and load the xclbin.
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernelName = vm["kernel"].as<std::string>();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  // batch1 must be the core's [1..8] accumulator contents.
  for (int i = 0; i < BATCH; i++) {
    uint32_t expected = static_cast<uint32_t>(i + 1);
    if (bufOut[i] != expected) {
      std::cout << "batch1[" << i << "] = " << bufOut[i]
                << " != " << expected << "\n";
      errors++;
    }
  }

  // batch2 must be all zero: aiex.buffer_clear's effect, read back without any
  // further involvement from the compute core.
  for (int i = 0; i < BATCH; i++) {
    if (bufOut[BATCH + i] != 0) {
      std::cout << "batch2[" << i << "] = " << bufOut[BATCH + i]
                << " != 0\n";
      errors++;
    }
  }

  std::cout << "batch1[0] = " << bufOut[0] << ", batch2[0] = " << bufOut[BATCH]
            << "\n";

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
