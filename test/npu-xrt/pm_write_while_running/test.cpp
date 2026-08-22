//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the program-memory-write-while-running experiment. The single
// runtime sequence collects two batches into one 16-element buffer, each the
// return value of sel_a() fanned across 8 words:
//   round0 = arg0[0:8]   before the program-memory write, always 7
//   round1 = arg0[8:16]  after it, 9 if the write took effect
//
// The expected round1 value is read from PM_EXPECT_ROUND1 so the same binary
// serves the negative control (variant A, expects 7) and the write variants
// (B-F, expect 9). See README.md.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int BATCH = 8;
constexpr int OUT_SIZE = 2 * BATCH; // round0 || round1
constexpr uint32_t SEL_A = 7;
constexpr uint32_t SEL_B = 9;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("pm_write_while_running");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  uint32_t expect1 = SEL_B;
  if (const char *e = std::getenv("PM_EXPECT_ROUND1"))
    expect1 = static_cast<uint32_t>(std::atoi(e));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

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
    // A hang here is itself a result: it means the program-memory write stalled
    // the core's instruction fetch permanently.
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut = bo_out.map<uint32_t *>();

  int errors = 0;

  // Each round must be internally uniform: the core fans one sel_a() result
  // across the whole buffer, so a mixed batch means a torn read, not a
  // half-applied patch.
  uint32_t r0 = bufOut[0];
  uint32_t r1 = bufOut[BATCH];
  for (int i = 0; i < BATCH; i++) {
    if (bufOut[i] != r0) {
      std::cout << "round0[" << i << "] = " << bufOut[i] << " != " << r0
                << "\n";
      errors++;
    }
    if (bufOut[BATCH + i] != r1) {
      std::cout << "round1[" << i << "] = " << bufOut[BATCH + i] << " != " << r1
                << "\n";
      errors++;
    }
  }

  std::cout << "round0 = " << r0 << " (sel_a, unpatched)\n";
  std::cout << "round1 = " << r1 << " (expected " << expect1 << ")\n";

  if (r0 != SEL_A) {
    std::cout << "round0 must be " << SEL_A
              << ": the design is broken independently of the patch\n";
    errors++;
  }

  if (r1 != expect1) {
    if (r1 == SEL_A)
      std::cout << "program memory write did not take effect\n";
    else if (r1 == SEL_B)
      std::cout << "program memory was patched when it should not have been\n";
    else
      std::cout << "round1 is neither " << SEL_A << " nor " << SEL_B
                << ": the write landed partially or corrupted the line\n";
    errors++;
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
