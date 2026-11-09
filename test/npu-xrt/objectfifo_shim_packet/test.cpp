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
  cxxopts::Options options("objectfifo_shim_packet");
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
  std::vector<xrt::bo> bos;
  for (int i = 0; i < 4; i++)
    bos.emplace_back(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(3 + i));

  std::vector<int32_t> a(N), b(N);
  for (int i = 0; i < N; i++) {
    a[i] = i + 1;
    b[i] = 3 * i + 7;
  }
  std::memcpy(bos[0].map<int32_t *>(), a.data(), N * sizeof(int32_t));
  std::memcpy(bos[1].map<int32_t *>(), b.data(), N * sizeof(int32_t));
  for (int i = 2; i < 4; i++)
    std::memset(bos[i].map<int32_t *>(), 0xff, N * sizeof(int32_t));
  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  for (auto &bo : bos)
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bos[0], bos[1], bos[2], bos[3]);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  int errors = 0;
  for (int k = 0; k < 2; k++) {
    bos[2 + k].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int32_t *out = bos[2 + k].map<int32_t *>();
    for (int i = 0; i < N; i++) {
      int32_t ref = k == 0 ? a[i] + 1 : b[i] + 2;
      if (out[i] != ref) {
        if (errors < 10)
          std::cout << "Error in output " << k << "[" << i << "]: " << out[i]
                    << " != " << ref << "\n";
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
