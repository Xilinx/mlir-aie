//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the core-reset reproducer. The single runtime sequence collects
// two output batches into one 16-element buffer: arg0[0:8] before the reset
// pulse and arg0[8:16] after it. The core writes a data-memory counter into the
// buffer on each run and increments it, so:
//   batch1 = N   (the value at the core's first run)
//   batch2 = N+1 (only produced if the reset restarted the core's PC)
// The counter's absolute start value is unspecified, so the proof the core
// re-ran is batch2 == batch1 + 1.

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
  cxxopts::Options options("local_reset_core");
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
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

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

  // Each batch must be internally uniform (the core fills the whole buffer with
  // one counter value).
  uint32_t b1 = bufOut[0];
  uint32_t b2 = bufOut[BATCH];
  for (int i = 0; i < BATCH; i++) {
    if (bufOut[i] != b1) {
      std::cout << "batch1[" << i << "] = " << bufOut[i] << " != " << b1
                << "\n";
      errors++;
    }
    if (bufOut[BATCH + i] != b2) {
      std::cout << "batch2[" << i << "] = " << bufOut[BATCH + i] << " != " << b2
                << "\n";
      errors++;
    }
  }

  std::cout << "batch1 = " << b1 << ", batch2 = " << b2 << "\n";

  // The proof the core restarted from a clean PC: it ran a second time and
  // emitted the incremented counter.
  if (b2 != b1 + 1) {
    std::cout
        << "Core did not re-run after reset: expected batch2 == batch1 + 1"
        << "\n";
    errors++;
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
