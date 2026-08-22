//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the program-memory overlay example. One core runs three
// different kernels in turn, each written into its program memory just before
// it runs, so the output rows can only all be right if all three were really
// loaded and executed.
//
//   row 0   out = in + 100   (the overlay calls back into the resident for 100)
//   row 1   out = in * 3
//   row 2   out = -in

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int N_ELEMS = 16;
constexpr int N_PHASES = 3;

static int32_t expected(int phase, int32_t in) {
  switch (phase) {
  case 0:
    return in + 100;
  case 1:
    return in * 3;
  default:
    return -in;
  }
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("program_memory_overlay");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernelName = vm["kernel"].as<std::string>();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N_ELEMS * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N_PHASES * N_ELEMS * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int32_t *bufIn = bo_in.map<int32_t *>();
  for (int i = 0; i < N_ELEMS; i++)
    bufIn[i] = i - 5; // straddle zero so the negate phase is not a no-op
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *bufOut = bo_out.map<int32_t *>();

  int errors = 0;
  for (int phase = 0; phase < N_PHASES; phase++) {
    for (int i = 0; i < N_ELEMS; i++) {
      int32_t got = bufOut[phase * N_ELEMS + i];
      int32_t want = expected(phase, bufIn[i]);
      if (got == want)
        continue;
      if (errors < 8)
        std::cout << "phase " << phase << " [" << i << "]: got " << got
                  << ", expected " << want << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\n" << errors << " errors -- an overlay did not take effect\n";
  std::cout << "failed.\n\n";
  return 1;
}
