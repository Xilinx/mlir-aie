//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the program-memory-write-while-running experiment. The single
// runtime sequence collects two rounds into one 16-element buffer. Each round
// is two halves: the first four words are sel_near_a()'s result, the last four
// are sel_far_a()'s. Both read 7 unpatched, and 9 where the write took effect.
//
// A variant patches exactly one of the two pairs, so the untouched half is a
// per-run control. PM_EXPECT_NEAR1 / PM_EXPECT_FAR1 say which to expect; both
// default to 7 (unpatched). See README.md.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Must match PAIR_DISTANCES in overlay_elf.py.
constexpr int DISTANCES[] = {64,   384,  512,  640,  768,  896,  960,
                             1024, 1152, 1280, 1408, 2048, 4160, 8320};
constexpr int BATCH = sizeof(DISTANCES) / sizeof(DISTANCES[0]);
constexpr int ROUNDS = 3;
constexpr int OUT_SIZE = ROUNDS * BATCH;
constexpr uint32_t SEL_A = 7;
constexpr uint32_t SEL_B = 9;

static void print_round(const uint32_t *out, int round) {
  std::cout << "round" << round << ":";
  for (int i = 0; i < BATCH; i++)
    std::cout << "  d" << DISTANCES[i] << "=" << out[round * BATCH + i];
  std::cout << "\n";
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("pm_write_while_running");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  // The pair this run patches, named by distance; every other pair is a
  // control.
  const char *distEnv = std::getenv("PM_PATCHED_DIST");
  int patchedDist = distEnv ? std::atoi(distEnv) : -1;
  int patched = -1;
  for (int i = 0; i < BATCH; i++)
    if (DISTANCES[i] == patchedDist)
      patched = i;
  if (patchedDist >= 0 && patched < 0) {
    std::cout << "PM_PATCHED_DIST=" << patchedDist
              << " is not one of the sweep distances\n";
    return 1;
  }

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

  for (int r = 0; r < ROUNDS; r++)
    print_round(bufOut, r);

  for (int i = 0; i < BATCH; i++) {
    if (bufOut[i] != SEL_A) {
      std::cout << "round0 d" << DISTANCES[i] << " = " << bufOut[i] << ", must "
                << "be " << SEL_A
                << ": the design is broken independently of the patch\n";
      errors++;
    }
    // Rounds 1 and 2 must both show the patch: a write that took effect stays
    // in effect, and one that never landed never appears.
    for (int r = 1; r < ROUNDS; r++) {
      uint32_t got = bufOut[r * BATCH + i];
      uint32_t want = (i == patched) ? SEL_B : SEL_A;
      if (got == want)
        continue;
      std::cout << "round" << r << " d" << DISTANCES[i] << ": ";
      if (got == SEL_A)
        std::cout << "program memory write did not take effect\n";
      else if (got == SEL_B)
        std::cout
            << "program memory was patched when it should not have been\n";
      else
        std::cout << "got " << got << ", neither " << SEL_A << " nor " << SEL_B
                  << ": the write landed partially or corrupted the line\n";
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
