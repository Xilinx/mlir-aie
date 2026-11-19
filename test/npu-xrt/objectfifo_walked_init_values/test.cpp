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

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int OBJECTS = 30;
constexpr int OBJECT_WORDS = 16;

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
  auto bo_out = xrt::bo(device, OBJECTS * OBJECT_WORDS * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int32_t *out = bo_out.map<int32_t *>();
  std::fill(out, out + OBJECTS * OBJECT_WORDS, -1);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Every third object is a splat; the others count up across the pool.
  int errors = 0;
  for (int i = 0; i < OBJECTS; i++)
    for (int k = 0; k < OBJECT_WORDS; k++) {
      int32_t ref = i % 3 == 2 ? 1000 + i : i * OBJECT_WORDS + k + 1;
      int32_t got = out[i * OBJECT_WORDS + k];
      if (got != ref && errors++ < 16)
        std::cout << "object " << i << " word " << k << ": " << got
                  << " != " << ref << "\n";
    }
  if (errors) {
    std::cout << "\nfailed with " << errors << " errors\n";
    return 1;
  }
  std::cout << "\nPASS!\n\n";
  return 0;
}
