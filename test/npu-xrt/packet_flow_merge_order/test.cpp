//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Asserts the EXACT arrival order of three merged packets, which is only
// achievable because the shim arbiter is in deterministic merge mode. See
// aie.mlir for why 2,1,3 is the expected order and 1,3,2 is what free
// arbitration produces.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int CHUNK = 256;
constexpr int NUM_SRC = 3;
constexpr int OUT_SIZE = CHUNK * NUM_SRC;

// The programmed merge-slot order: East:2 (payload 2), North:2 (payload 1),
// East:0 (payload 3). Free arbitration on this part yields {1, 3, 2}.
constexpr int8_t kExpectedOrder[NUM_SRC] = {2, 1, 3};
constexpr int8_t kFreeArbitrationOrder[NUM_SRC] = {1, 3, 2};

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  auto device = xrt::device(0);
  xrt::xclbin xclbin(std::string("aie.xclbin"));
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind("MLIR_AIE", 0) == 0;
      });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  int8_t *out = bo_out.map<int8_t *>();
  std::fill(out, out + OUT_SIZE, static_cast<int8_t>(-1));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3u, bo_instr, instr_v.size(), bo_out);
  // Bounded wait purely so a broken build reports instead of hanging the
  // harness. This design cannot deadlock: every packet has a destination BD.
  if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
    std::cout << "Kernel timed out.\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Each source contributes one contiguous 256-byte block; the block order is
  // what deterministic merge fixes.
  int errors = 0;
  std::cout << "arrival order: ";
  for (int b = 0; b < NUM_SRC; b++)
    std::cout << static_cast<int>(out[b * CHUNK]) << " ";
  std::cout << "(expected 2 1 3)\n";

  for (int b = 0; b < NUM_SRC; b++) {
    for (int i = 0; i < CHUNK; i++) {
      int8_t got = out[b * CHUNK + i];
      if (got != kExpectedOrder[b]) {
        if (errors < 8)
          std::cout << "byte " << (b * CHUNK + i) << ": got "
                    << static_cast<int>(got) << " want "
                    << static_cast<int>(kExpectedOrder[b]) << "\n";
        errors++;
      }
    }
  }

  if (errors) {
    bool looksUnscheduled = true;
    for (int b = 0; b < NUM_SRC; b++)
      if (out[b * CHUNK] != kFreeArbitrationOrder[b])
        looksUnscheduled = false;
    if (looksUnscheduled)
      std::cout << "\nThe order is 1 3 2, which is what this design produces "
                   "with free arbitration. The deterministic merge schedule "
                   "was not applied -- check that the `deterministic_merge` "
                   "attribute survived compilation and reached "
                   "XAie_StrmSwDeterministicMergeConfig.\n";
    std::cout << "\nfailed. " << errors << " mismatched bytes.\n";
    return 1;
  }

  std::cout << "\nPASS!\n";
  return 0;
}
