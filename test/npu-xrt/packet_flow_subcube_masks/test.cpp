//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the relaxed-mask packet routing test. Four packet ids share one
// shim source and carry distinguishable payloads (id N sends 1000 + 100*N + i),
// so a mask that claims one id too many shows up as the wrong payload rather
// than as a missing transfer. Ids 0, 3, 4 land in memtile a0, a1, a2 in that
// order and id 1 in b0. See aie.mlir / README.md.

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int SLICE = 8;
// Send order, which is also the order the receive buffers fill.
constexpr int A_IDS[3] = {0, 3, 4};
constexpr int B_ID = 1;

static int32_t payload(int id, int i) { return 1000 + 100 * id + i; }

int main(int argc, const char *argv[]) {
  cxxopts::Options options("packet_flow_subcube_masks");
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
  auto bo_in = xrt::bo(device, 4 * SLICE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_oa = xrt::bo(device, 3 * SLICE * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_ob = xrt::bo(device, SLICE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(5));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int32_t *in = bo_in.map<int32_t *>();
  int32_t *oa = bo_oa.map<int32_t *>();
  int32_t *ob = bo_ob.map<int32_t *>();
  for (int s = 0; s < 3; s++)
    for (int i = 0; i < SLICE; i++)
      in[s * SLICE + i] = payload(A_IDS[s], i);
  for (int i = 0; i < SLICE; i++)
    in[3 * SLICE + i] = payload(B_ID, i);
  for (int i = 0; i < 3 * SLICE; i++)
    oa[i] = 0xdead;
  for (int i = 0; i < SLICE; i++)
    ob[i] = 0xdead;
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_oa.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ob.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_oa, bo_ob);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "kernel did not complete\n";
    return 1;
  }
  bo_oa.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int s = 0; s < 3; s++)
    for (int i = 0; i < SLICE; i++) {
      int32_t want = payload(A_IDS[s], i);
      if (oa[s * SLICE + i] != want) {
        std::cout << "packet id " << A_IDS[s] << " slot " << i << ": got "
                  << oa[s * SLICE + i] << " want " << want << "\n";
        errors++;
      }
    }
  for (int i = 0; i < SLICE; i++) {
    int32_t want = payload(B_ID, i);
    if (ob[i] != want) {
      std::cout << "packet id " << B_ID << " slot " << i << ": got " << ob[i]
                << " want " << want << "\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
