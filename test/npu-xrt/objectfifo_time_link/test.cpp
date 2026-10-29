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

constexpr int OBJ = 16;
constexpr int OBJECTS = 8;
constexpr int N = OBJ * OBJECTS;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_time_link");
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
  auto bo_in = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  std::vector<int32_t> in(N);
  for (int i = 0; i < N; i++)
    in[i] = i;
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
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *out = bo_out.map<int32_t *>();

  // Object k went to core a when k is even (+1000) and to core b when odd
  // (+2000). Each returned object must be exactly one such result, and every
  // input object must come back once.
  std::vector<bool> seen(OBJECTS, false);
  int errors = 0;
  for (int o = 0; o < OBJECTS; o++) {
    const int32_t *obj = out + o * OBJ;
    int32_t offset = obj[0] >= 2000 ? 2000 : 1000;
    int k = (obj[0] - offset) / OBJ;
    bool ok = k >= 0 && k < OBJECTS && !seen[k] &&
              offset == (k % 2 == 0 ? 1000 : 2000);
    for (int j = 0; ok && j < OBJ; j++)
      ok = obj[j] == in[k * OBJ + j] + offset;
    if (!ok) {
      std::cout << "Returned object " << o << " starting " << obj[0]
                << " matches no input object\n";
      errors++;
      continue;
    }
    seen[k] = true;
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
