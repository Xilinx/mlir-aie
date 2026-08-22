//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every byte the kernel stored must arrive. Reports the pattern of the ones
// that did not, because the pattern is the diagnosis: with the miscompile the
// missing bytes are exactly those at i % 4 == 3.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int N = 1024;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("peano_scalar_store_canary");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  auto instr = test_utils::load_instr_binary(vm["instr"].as<std::string>());
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);
  xrt::hw_context ctx(device, xclbin.get_uuid());
  auto k = xrt::kernel(ctx, vm["kernel"].as<std::string>());

  auto bo_instr = xrt::bo(device, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                          k.group_id(1));
  auto bo_out = xrt::bo(device, N, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  memcpy(bo_instr.map<void *>(), instr.data(), instr.size() * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bo_out.map<void *>(), 0, N);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = k(3, bo_instr, instr.size(), bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint8_t *p = bo_out.map<uint8_t *>();
  std::vector<int> by_mod4(4, 0);
  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++) {
    if (p[i] == 0x11)
      continue;
    if (first < 0)
      first = i;
    wrong++;
    by_mod4[i % 4]++;
  }

  if (!wrong) {
    std::cout << "all " << N << " bytes stored\nPASS!\n";
    return 0;
  }
  std::cout << wrong << " of " << N << " bytes never stored; first at " << first
            << "; by i%4 = [" << by_mod4[0] << "," << by_mod4[1] << ","
            << by_mod4[2] << "," << by_mod4[3] << "]\n";
  std::cout << "failed.\n";
  return 1;
}
