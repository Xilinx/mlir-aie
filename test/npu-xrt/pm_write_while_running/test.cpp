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

constexpr int BATCH = 8;
constexpr int HALF = BATCH / 2;     // [0,HALF) near pair, [HALF,BATCH) far pair
constexpr int OUT_SIZE = 2 * BATCH; // round0 || round1
constexpr uint32_t SEL_A = 7;
constexpr uint32_t SEL_B = 9;

// One half of one round: every word must agree (the core fans a single result
// across it, so a mixed half is a torn read, not a half-applied patch).
static uint32_t read_half(const uint32_t *out, int round, int base,
                          const char *what, int &errors) {
  const uint32_t *p = out + round * BATCH + base;
  for (int i = 1; i < HALF; i++)
    if (p[i] != p[0]) {
      std::cout << "round" << round << " " << what << "[" << i << "] = " << p[i]
                << " != " << p[0] << "\n";
      errors++;
    }
  return p[0];
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("pm_write_while_running");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  // Which half the variant patches; the other must stay untouched.
  auto expect = [](const char *var) {
    const char *e = std::getenv(var);
    return e ? static_cast<uint32_t>(std::atoi(e)) : SEL_A;
  };
  uint32_t expectNear = expect("PM_EXPECT_NEAR1");
  uint32_t expectFar = expect("PM_EXPECT_FAR1");

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

  uint32_t near0 = read_half(bufOut, 0, 0, "near", errors);
  uint32_t far0 = read_half(bufOut, 0, HALF, "far", errors);
  uint32_t near1 = read_half(bufOut, 1, 0, "near", errors);
  uint32_t far1 = read_half(bufOut, 1, HALF, "far", errors);

  std::cout << "round0 near = " << near0 << "  far = " << far0
            << "  (unpatched)\n";
  std::cout << "round1 near = " << near1 << "  far = " << far1
            << "  (expected near " << expectNear << ", far " << expectFar
            << ")\n";

  if (near0 != SEL_A || far0 != SEL_A) {
    std::cout << "round0 must be " << SEL_A
              << " in both halves: the design is broken independently of the "
                 "patch\n";
    errors++;
  }

  struct {
    const char *what;
    uint32_t got, want;
  } checks[] = {{"near", near1, expectNear}, {"far", far1, expectFar}};
  for (auto &c : checks) {
    if (c.got == c.want)
      continue;
    std::cout << c.what << " half: ";
    if (c.got == SEL_A)
      std::cout << "program memory write did not take effect\n";
    else if (c.got == SEL_B)
      std::cout << "program memory was patched when it should not have been\n";
    else
      std::cout << "got " << c.got << ", neither " << SEL_A << " nor " << SEL_B
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
