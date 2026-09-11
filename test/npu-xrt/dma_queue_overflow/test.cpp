//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifies every transfer landed: a dropped push is silent at the dispatch
// level, so only the data distinguishes success from a lost transfer.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int ELEMS = 2048;   // per transfer
constexpr int NPUSH = 14;     // pushes on one 4-deep channel
constexpr int BUFSZ = 65536;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("dma_queue_overflow");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instrV =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto it = std::find_if(xkernels.begin(), xkernels.end(),
                         [node](xrt::xclbin::kernel &k) {
                           return k.get_name().rfind(node, 0) == 0;
                         });
  if (it == xkernels.end()) {
    std::cout << "Kernel '" << node << "' not found in xclbin\n";
    return 1;
  }
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, it->get_name());

  auto boInstr = xrt::bo(device, instrV.size() * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto boA = xrt::bo(device, BUFSZ * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(3));
  auto boOut = xrt::bo(device, BUFSZ * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(4));

  std::memcpy(boInstr.map<void *>(), instrV.data(),
              instrV.size() * sizeof(int));
  auto *a = boA.map<int32_t *>();
  for (int i = 0; i < BUFSZ; ++i)
    a[i] = i;
  auto *out = boOut.map<int32_t *>();
  std::memset(out, 0, BUFSZ * sizeof(int32_t));

  boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boOut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, boInstr, instrV.size(), boA, boOut);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete, returned status: " << r << "\n";
    return 1;
  }
  boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Every transfer sends buf_a[0..ELEMS), so each of the NPUSH output slices
  // must match it. A dropped push leaves its slice zeroed.
  int errors = 0;
  for (int t = 0; t < NPUSH && errors < 10; ++t)
    for (int i = 0; i < ELEMS; ++i)
      if (out[t * ELEMS + i] != a[i]) {
        std::cout << "transfer " << t << " element " << i << ": got "
                  << out[t * ELEMS + i] << " expected " << a[i] << "\n";
        if (++errors >= 10)
          break;
      }

  if (!errors) {
    std::cout << "PASS! (" << NPUSH << " pushes on a 4-deep queue, all landed)"
              << std::endl;
    return 0;
  }
  std::cout << "\nfailed.\n" << std::endl;
  return 1;
}
