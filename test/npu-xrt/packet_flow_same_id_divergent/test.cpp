//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host check for two DIVERGENT flows that share packet id 0.
//
// Buffer A is fed by tile(0,2) (sentinel 1); buffer B by tile(1,2) (sentinel
// 2). Because the two flows never merge, the result is deterministic: A must be
// all 1s and B must be all 2s. Cross-contamination (a 2 appearing in A) is the
// signature of two same-id streams having shared a link and then been split
// again -- the failure mode fixed by PR #3472.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int CHUNK = 256;

static int check(const char *name, int8_t *buf, int8_t want) {
  int errors = 0, contaminated = 0;
  for (int i = 0; i < CHUNK; i++) {
    if (buf[i] != want) {
      if (errors < 4)
        std::cout << "  " << name << "[" << i << "] = " << (int)buf[i]
                  << ", expected " << (int)want << "\n";
      if (buf[i] != -1)
        contaminated++;
      errors++;
    }
  }
  if (errors)
    std::cout << "  " << name << ": " << errors << " mismatches ("
              << contaminated << " carrying another source's payload)\n";
  else
    std::cout << "  " << name << ": all " << CHUNK << " bytes == " << (int)want
              << "  OK\n";
  return errors;
}

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  auto device = xrt::device(0);
  std::string xclbin_name = (argc > 1) ? argv[1] : "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";

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
  auto bo_a = xrt::bo(device, CHUNK * sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_b = xrt::bo(device, CHUNK * sizeof(int8_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  int8_t *bufA = bo_a.map<int8_t *>();
  int8_t *bufB = bo_b.map<int8_t *>();
  std::fill(bufA, bufA + CHUNK, (int8_t)-1);
  std::fill(bufB, bufB + CHUNK, (int8_t)-1);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  try {
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b);
    if (run.wait2(std::chrono::seconds(10)) == std::cv_status::timeout) {
      std::cout
          << "Kernel TIMED OUT after 10s -- stream stalled.\n\nfailed.\n\n";
      return 1;
    }
    ert_cmd_state r = run.state();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Status: " << r
                << "\n\nfailed.\n\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cout << "Kernel dispatch failed: " << e.what() << "\n\nfailed.\n\n";
    return 1;
  }

  bo_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "destination buffers:\n";
  int errors = check("A(shim DMA:0, from tile 0,2)", bufA, 1) +
               check("B(shim DMA:1, from tile 1,2)", bufB, 2);

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed. errors=" << errors << "\n\n";
  return 1;
}
