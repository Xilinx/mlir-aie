// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#ifndef XCLBIN
#define XCLBIN "final.xclbin"
#endif

#ifndef INSTS_TXT
#define INSTS_TXT "insts.bin"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define DTYPE int32_t
#define A_DATATYPE DTYPE
#define B_DATATYPE DTYPE

#define BUFFER_LEN 65536
#define BUFFER_TILE_LEN (BUFFER_LEN >> 5)
#define A_LEN BUFFER_LEN
#define B_LEN BUFFER_LEN

#define A_SIZE (A_LEN * sizeof(A_DATATYPE)) // in bytes
#define B_SIZE (B_LEN * sizeof(B_DATATYPE)) // in bytes

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(INSTS_TXT);
  assert(instr_v.size() > 0);

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);

  // Get the kernel from the xclbin
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), KERNEL_NAME) == 0);

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  A_DATATYPE *buf_a = bo_a.map<A_DATATYPE *>();
  for (int i = 0; i < A_SIZE / sizeof(buf_a[0]); i++) {
    buf_a[i] = i; // consecutive ints
  }
  B_DATATYPE *buf_b = bo_b.map<B_DATATYPE *>();
  memset(buf_b, 0, B_SIZE); // zeroes, becuase this is output

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3; // TODO: should be 2?
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  /*
  for (int i = 0; i < B_SIZE / sizeof(buf_b[0]); i++) {
    std::cout << std::setw(4) << (long)buf_b[i] << " ";
  }
  std::cout << std::endl;
  */

  bool same = true;
  for (DTYPE i = 0; (i < B_SIZE / sizeof(buf_b[0])) && same; i++) {
    same = buf_b[i] == i + 1;
    if (!same) {
      std::cout << "(" << i << ") (" << BUFFER_TILE_LEN << ") Expected "
                << (i % BUFFER_TILE_LEN) + 1 << " got " << buf_b[i]
                << std::endl;
    }
  }

  if (same) {
    std::cout << "PASS!" << std::endl;
  } else {
    std::cout << "FAIL." << std::endl;
    return 1;
  }

  return 0;
}
