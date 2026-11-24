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
constexpr int STEPS = 8;
constexpr int IN_LEN = OBJ * STEPS;
constexpr int C_LEN = OBJ * STEPS;
constexpr int OUT_LEN = 2 * OBJ * STEPS;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("objectfifo_channel_turns_join");
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
  auto bo_a = xrt::bo(device, IN_LEN * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_c = xrt::bo(device, C_LEN * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_LEN * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  std::vector<int32_t> a(IN_LEN);
  for (int i = 0; i < IN_LEN; i++)
    a[i] = 5 * i - 300;
  std::memcpy(bo_a.map<int32_t *>(), a.data(), IN_LEN * sizeof(int32_t));
  std::memset(bo_c.map<int32_t *>(), 0xff, C_LEN * sizeof(int32_t));
  std::memset(bo_out.map<int32_t *>(), 0xff, OUT_LEN * sizeof(int32_t));
  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_c, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int32_t *cout = bo_c.map<int32_t *>();
  int32_t *out = bo_out.map<int32_t *>();

  // Step s: core r reads a's object s and sends x = a + 1, y = 2 * a and
  // c = 5 * a - 7; core s gets big = [x, y] and writes 3 * big + 1.
  int errors = 0;
  auto check = [&](const char *what, int i, int32_t got, int32_t want) {
    if (got != want && errors++ < 10)
      std::cout << what << " " << i << ": got " << got << ", want " << want
                << "\n";
  };
  for (int s = 0; s < STEPS; s++)
    for (int j = 0; j < OBJ; j++) {
      int32_t v = a[s * OBJ + j];
      check("c", s * OBJ + j, cout[s * OBJ + j], 5 * v - 7);
      check("out", 2 * s * OBJ + j, out[2 * s * OBJ + j], 3 * (v + 1) + 1);
      check("out", 2 * s * OBJ + OBJ + j, out[2 * s * OBJ + OBJ + j],
            3 * (2 * v) + 1);
    }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed: " << errors << " mismatches.\n\n";
  return 1;
}
