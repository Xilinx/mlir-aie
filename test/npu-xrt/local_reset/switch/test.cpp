//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the stream-switch-reset test. The resident buffer holds
// [100..107]; the runtime sequence config-disables then re-enables the tile's
// DMA:0 -> South switch connection and re-arms the MM2S lock each dispatch. We
// dispatch the same kernel many times and require every dispatch to return the
// resident buffer unchanged -- i.e. the disable-and-re-enable protocol keeps the
// run-forever routed channel correct across dispatches.

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int SIZE = 8;
constexpr int DISPATCHES = 8;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("local_reset_switch");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int errors = 0;
  for (int d = 0; d < DISPATCHES; d++) {
    // Sentinel-fill the output so a hung or partial transfer reads as garbage
    // rather than as a valid (possibly stale-correct) result.
    uint32_t *out = bo_out.map<uint32_t *>();
    for (int i = 0; i < SIZE; i++)
      out[i] = 0xdeadbeef;
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, bo_instr, instr_v.size(), bo_out);
    if (run.wait() != ERT_CMD_STATE_COMPLETED) {
      std::cout << "dispatch " << d << ": kernel did not complete\n";
      return 1;
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < SIZE; i++) {
      uint32_t want = 100 + i;
      if (out[i] != want) {
        std::cout << "dispatch " << d << " out[" << i << "] = " << out[i]
                  << " != " << want << "\n";
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
