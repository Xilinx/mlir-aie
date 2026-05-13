//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int N = 32;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("npu2_buffer_alignment_2579");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(),
      [Node](xrt::xclbin::kernel &k) { return k.get_name().rfind(Node, 0) == 0; });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(int));

  int32_t *in_p = bo_in.map<int32_t *>();
  for (int i = 0; i < N; i++) in_p[i] = i * 7 + 1;

  int32_t *out_p = bo_out.map<int32_t *>();
  for (int i = 0; i < N; i++) out_p[i] = -1;

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int i = 0; i < N; i++) {
    int32_t expected = in_p[i] + 1;
    if (out_p[i] != expected) {
      if (errors < 8)
        std::cerr << "[" << i << "] in=" << in_p[i] << " want=" << expected
                  << " got=" << out_p[i] << "\n";
      errors++;
    }
  }
  if (errors == 0) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed: " << errors << "/" << N << " mismatches\n\n";
  return 1;
}
