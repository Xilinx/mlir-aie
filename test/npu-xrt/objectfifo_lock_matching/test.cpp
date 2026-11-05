//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int SIZE = 16;
constexpr int32_t STAGES = 24;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_lock_matching");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(),
      [&](xrt::xclbin::kernel &k) { return k.get_name().rfind(node, 0) == 0; });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto bo_out = xrt::bo(device, SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  std::vector<int32_t> in(SIZE);
  for (int i = 0; i < SIZE; i++)
    in[i] = 1000 * i - 7;
  memcpy(bo_in.map<int32_t *>(), in.data(), SIZE * sizeof(int32_t));
  memset(bo_out.map<int32_t *>(), 0, SIZE * sizeof(int32_t));
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete.\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  const int32_t *out = bo_out.map<int32_t *>();
  int errors = 0;
  for (int i = 0; i < SIZE; i++) {
    int32_t ref = in[i] + STAGES;
    if (out[i] != ref) {
      std::cout << "Error at " << i << ": " << out[i] << " != " << ref << "\n";
      errors++;
    }
  }
  std::cout << (errors ? "\nfailed.\n\n" : "\nPASS!\n\n");
  return errors ? 1 : 0;
}
