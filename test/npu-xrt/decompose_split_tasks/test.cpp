// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

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

#define PAIRS 17393
#define STRIDE 3
#define LEN 65536
#define SIZE (LEN * sizeof(int32_t))

int main() {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  xrt::device device = xrt::device(0);
  xrt::xclbin xclbin = xrt::xclbin(std::string("final.xclbin"));
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind("MLIR_AIE", 0) == 0;
      });
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, xkernel.get_name());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  int32_t *buf_a = bo_a.map<int32_t *>();
  for (int i = 0; i < LEN; i++)
    buf_a[i] = i;
  int32_t *buf_b = bo_b.map<int32_t *>();
  std::fill(buf_b, buf_b + LEN, -1);
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // The first two elements of each STRIDE are copied; the rest are untouched.
  int errors = 0;
  for (int i = 0; i < LEN; i++) {
    bool moved = i / STRIDE < PAIRS && i % STRIDE < 2;
    int32_t expected = moved ? i : -1;
    if (buf_b[i] != expected) {
      if (errors < 10)
        std::cout << "element " << i << ": expected " << expected << " got "
                  << buf_b[i] << "\n";
      errors++;
    }
  }
  if (errors) {
    std::cout << "FAIL: " << errors << " mismatches\n";
    return 1;
  }
  std::cout << "PASS!\n";
  return 0;
}
