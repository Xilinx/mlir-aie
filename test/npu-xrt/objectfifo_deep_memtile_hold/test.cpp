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

constexpr int ROW = 64;
constexpr int ROWS = 80;
constexpr int IN = ROW * ROWS;
constexpr int OUT = ROW * ROWS;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_deep_memtile_hold");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(node, 0) == 0;
                               });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, IN * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_bc = xrt::bo(device, IN * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  std::vector<int32_t> a(IN), bc(IN, 0);
  for (int i = 0; i < IN; i++)
    a[i] = 7 * i - 3;
  std::memcpy(bo_a.map<int32_t *>(), a.data(), IN * sizeof(int32_t));
  std::memcpy(bo_bc.map<int32_t *>(), bc.data(), IN * sizeof(int32_t));
  std::memset(bo_out.map<int32_t *>(), 0xff, OUT * sizeof(int32_t));
  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_bc, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *out = bo_out.map<int32_t *>();

  // Row r comes back with r added to each element.
  int errors = 0;
  for (int r = 0; r < ROWS; r++)
    for (int j = 0; j < ROW; j++) {
      int32_t want = a[r * ROW + j] + r;
      if (out[r * ROW + j] != want && errors++ < 10)
        std::cout << "row " << r << " element " << j << ": got "
                  << out[r * ROW + j] << ", want " << want << "\n";
    }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed: " << errors << " mismatches.\n\n";
  return 1;
}
