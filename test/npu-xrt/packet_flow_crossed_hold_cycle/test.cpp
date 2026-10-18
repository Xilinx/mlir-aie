//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int N = 256;
constexpr int PACKETS = 8;

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");
  auto device = xrt::device(0);
  xrt::xclbin xclbin(std::string("aie.xclbin"));
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind("MLIR_AIE", 0) == 0;
                               });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_o = xrt::bo(device, 2 * N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));

  int32_t *o = bo_o.map<int32_t *>();
  std::fill(o, o + 2 * N, -1);
  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_o.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_o);
  ert_cmd_state r = run.wait(std::chrono::milliseconds(10000));
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "HANG: kernel did not complete, state " << r << "\n";
    return 2;
  }
  bo_o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int errors = 0;
  for (int i = 0; i < 2 * N; i++) {
    int32_t base = i < N ? 1000 : 5000;
    int32_t want = 0;
    for (int p = 0; p < PACKETS; p++)
      want += base + p * 7919 + (i % N) * 31;
    if (o[i] != want && errors++ < 5)
      std::cout << "o[" << i << "] = " << o[i] << " != " << want << "\n";
  }
  std::cout << (errors ? "FAIL" : "PASS") << " (" << errors << " errors)\n";
  return errors ? 1 : 0;
}
