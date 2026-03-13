//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// End-to-end test for multi-source packet flows.
//
// Two AIE cores (tile 0,2 and tile 0,3) each produce 8 int32 values and send
// them to the shim via a single packet_flow with two aie.packet_source ops
// (fan-in). The host verifies that data from BOTH tiles arrives correctly.
//
// tile(0,2) produces: [1, 2, 3, 4, 5, 6, 7, 8]
// tile(0,3) produces: [101, 102, 103, 104, 105, 106, 107, 108]
//
// Expected output (sorted): [1, 2, 3, 4, 5, 6, 7, 8, 101, ..., 108]
//
// With the pathfinder bug, only tile(0,3) is routed; tile(0,2)'s DMA stalls
// and the host times out waiting for 16 elements.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int OUT_NELEMS = 16;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("multi_source_packet_flow");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel: " << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  // dummy_a and dummy_b are unused by the device; allocated to satisfy the
  // 3-BO kernel interface (group_id 3, 4, 5).
  auto bo_dummy_a = xrt::bo(device, 8 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(3));
  auto bo_dummy_b = xrt::bo(device, 8 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_NELEMS * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_dummy_a, bo_dummy_b, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *bufOut = bo_out.map<int32_t *>();

  // Collect and sort output. The relative arrival order of the two tiles'
  // packets is non-deterministic, so we verify by sorted comparison.
  std::vector<int32_t> got(bufOut, bufOut + OUT_NELEMS);
  // Sort manually to avoid depending on <algorithm>, which some toolchains
  // don't expose through the clang driver used in lit tests.
  for (int i = 0; i < OUT_NELEMS; i++)
    for (int j = i + 1; j < OUT_NELEMS; j++)
      if (got[i] > got[j])
        std::swap(got[i], got[j]);

  // tile(0,2) produces [1..8], tile(0,3) produces [101..108].
  std::vector<int32_t> expected;
  for (int32_t i = 1; i <= 8; i++)
    expected.push_back(i);
  for (int32_t i = 101; i <= 108; i++)
    expected.push_back(i);

  int errors = 0;
  for (int i = 0; i < OUT_NELEMS; i++) {
    if (got[i] != expected[i]) {
      std::cout << "Error at sorted position " << i << ": got " << got[i]
                << ", expected " << expected[i] << "\n";
      errors++;
    } else if (verbosity >= 1) {
      std::cout << "Correct at sorted position " << i << ": " << got[i] << "\n";
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
