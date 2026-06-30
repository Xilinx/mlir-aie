//===- test.cpp -------------------------------------------000---*- C++ -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 16-host-buffer XRT test: allocates 8 input + 8 output BOs, fills each input
// with a distinct pattern (i * (pair + 1)), DMA-passthrough through shim, and
// verifies each output matches its input.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int BUF_SIZE = 64;
constexpr int NUM_PAIRS = 8;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("many_buffers");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernel_name = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [&](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(kernel_name, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // BOs are at group_id 3..18 (opcode=0, instr=1, ninstr=2, then 16 BOs)
  std::vector<xrt::bo> bo_in(NUM_PAIRS), bo_out(NUM_PAIRS);
  for (int p = 0; p < NUM_PAIRS; p++) {
    bo_in[p] = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3 + p * 2));
    bo_out[p] = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4 + p * 2));
  }

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (int p = 0; p < NUM_PAIRS; p++) {
    int32_t *buf = bo_in[p].map<int32_t *>();
    for (int i = 0; i < BUF_SIZE; i++)
      buf[i] = i * (p + 1);
    bo_in[p].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in[0], bo_out[0],
                    bo_in[1], bo_out[1], bo_in[2], bo_out[2], bo_in[3],
                    bo_out[3], bo_in[4], bo_out[4], bo_in[5], bo_out[5],
                    bo_in[6], bo_out[6], bo_in[7], bo_out[7]);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  int errors = 0;
  for (int p = 0; p < NUM_PAIRS; p++) {
    bo_out[p].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int32_t *out = bo_out[p].map<int32_t *>();
    for (int i = 0; i < BUF_SIZE; i++) {
      int32_t expected = i * (p + 1);
      if (out[i] != expected) {
        std::cout << "pair " << p << " idx " << i << ": got " << out[i]
                  << " expected " << expected << "\n";
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "PASS!\n";
    return 0;
  } else {
    std::cout << "failed.\n";
    return 1;
  }
}
