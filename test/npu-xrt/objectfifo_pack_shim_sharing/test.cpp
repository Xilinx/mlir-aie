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

constexpr int N = 64;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_pack_shim_sharing");
  test_utils::add_default_options(options);
  options.add_options()("instr-b", "member b's instructions",
                        cxxopts::value<std::string>())(
      "instr-c", "member c's instructions", cxxopts::value<std::string>());
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

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

  // Each member adds its own constant; run them one after another, as a pack
  // runs its members.
  const std::vector<std::pair<std::string, int32_t>> members = {
      {vm["instr"].as<std::string>(), 1},
      {vm["instr-b"].as<std::string>(), 2},
      {vm["instr-c"].as<std::string>(), 3}};
  int errors = 0;
  for (auto [path, add] : members) {
    std::vector<uint32_t> instr_v = test_utils::load_instr_binary(path);
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(3));
    auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(4));
    std::vector<int32_t> in(N);
    for (int i = 0; i < N; i++)
      in[i] = 7 * i + add;
    std::memcpy(bo_in.map<int32_t *>(), in.data(), N * sizeof(int32_t));
    std::memset(bo_out.map<int32_t *>(), 0xff, N * sizeof(int32_t));
    std::memcpy(bo_instr.map<void *>(), instr_v.data(),
                instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << path << " did not complete. Returned status: " << r << "\n";
      return 1;
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int32_t *out = bo_out.map<int32_t *>();
    for (int i = 0; i < N; i++) {
      if (out[i] != in[i] + add) {
        if (errors < 10)
          std::cout << path << "[" << i << "]: " << out[i]
                    << " != " << in[i] + add << "\n";
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
