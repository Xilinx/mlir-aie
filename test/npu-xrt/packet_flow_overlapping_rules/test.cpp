//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the overlapping-packet-rules test (Xilinx/mlir-aie#437).
//
// Two packet streams go up the same wire: x with packet id 10 (destined for
// memtile buf_a) and y with packet id 14 (destined for buf_b). The memtile's
// slave rules overlap, so the id-10 rule also matches 14 and, being first,
// claims it -- buf_b never receives anything and keeps its -1 sentinel.
//
// The assertion is on out_b only. out_a is printed, not checked: under the bug
// buf_a receives both streams, and which one it holds when the readback fires
// is a race. See aie.mlir / README.md.

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int BUF_SIZE = 8;
constexpr int32_t X_BASE = 100;
constexpr int32_t Y_BASE = 200;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("packet_flow_overlapping_rules");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernelName = vm["kernel"].as<std::string>();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_x = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_y = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_oa = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_ob = xrt::bo(device, BUF_SIZE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int32_t *x = bo_x.map<int32_t *>();
  int32_t *y = bo_y.map<int32_t *>();
  int32_t *oa = bo_oa.map<int32_t *>();
  int32_t *ob = bo_ob.map<int32_t *>();
  for (int i = 0; i < BUF_SIZE; i++) {
    x[i] = X_BASE + i;
    y[i] = Y_BASE + i;
    // Distinct from the memtile's -1 sentinel, so a transfer that never
    // happened is distinguishable from one that copied the sentinel out.
    oa[i] = 0xdead;
    ob[i] = 0xdead;
  }
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_oa.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_x, bo_y, bo_oa, bo_ob);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "kernel did not complete\n";
    return 1;
  }
  bo_oa.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "buf_a (packet id 10, not checked):";
  for (int i = 0; i < BUF_SIZE; i++)
    std::cout << ' ' << oa[i];
  std::cout << "\nbuf_b (packet id 14):";
  for (int i = 0; i < BUF_SIZE; i++)
    std::cout << ' ' << ob[i];
  std::cout << "\n";

  int errors = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    int32_t want = Y_BASE + i;
    if (ob[i] != want) {
      std::cout << "buf_b[" << i << "] = " << ob[i] << " != " << want << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\npacket id 14 did not reach its destination\n";
  std::cout << "\nfailed.\n\n";
  return 1;
}
