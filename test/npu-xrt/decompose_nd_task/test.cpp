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

#define LEN 32768
#define SIZE (LEN * sizeof(int32_t))
#define PASSES 2

// The fill's access pattern, outermost first (aie2.py's SIZES and STRIDES).
static const int sizes[6] = {2, 3, 4, 2, 8, 16};
static const int strides[6] = {16384, 4096, 16, 1024, 128, 1};

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

  // The drain holds the gathered elements in the order the pattern visits
  // them, innermost fastest, once per pass; the rest of B is untouched.
  std::vector<int32_t> expected;
  for (int p = 0; p < PASSES; p++)
    for (int i5 = 0; i5 < sizes[0]; i5++)
      for (int i4 = 0; i4 < sizes[1]; i4++)
        for (int i3 = 0; i3 < sizes[2]; i3++)
          for (int i2 = 0; i2 < sizes[3]; i2++)
            for (int i1 = 0; i1 < sizes[4]; i1++)
              for (int i0 = 0; i0 < sizes[5]; i0++)
                expected.push_back(i5 * strides[0] + i4 * strides[1] +
                                   i3 * strides[2] + i2 * strides[3] +
                                   i1 * strides[4] + i0 * strides[5]);
  expected.resize(LEN, -1);

  int errors = 0;
  for (int i = 0; i < LEN; i++) {
    if (buf_b[i] != expected[i]) {
      if (errors < 10)
        std::cout << "element " << i << ": expected " << expected[i] << " got "
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
