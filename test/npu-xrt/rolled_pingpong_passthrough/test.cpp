// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstring>
#include <fstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#ifndef XCLBIN
#define XCLBIN std::string("final.xclbin")
#endif

#ifndef INSTS_TXT
#define INSTS_TXT "insts.bin"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

#define DTYPE int32_t
#define TILE_LEN 256
#define N_TILES 8
#define IN_LEN TILE_LEN
#define OUT_LEN (TILE_LEN * N_TILES)

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(INSTS_TXT);
  assert(instr_v.size() > 0);

  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);

  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  xrt::xclbin::kernel xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), KERNEL_NAME) == 0);

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input = xrt::bo(device, IN_LEN * sizeof(DTYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output = xrt::bo(device, OUT_LEN * sizeof(DTYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  DTYPE *buf_input = bo_input.map<DTYPE *>();
  for (int i = 0; i < IN_LEN; i++)
    buf_input[i] = i + 1;

  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, OUT_LEN * sizeof(DTYPE));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input, bo_output);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Golden: output is N_TILES copies of input tile.
  bool pass = true;
  for (int tile = 0; tile < N_TILES; tile++) {
    for (int i = 0; i < TILE_LEN; i++) {
      DTYPE expected = buf_input[i];
      DTYPE got = buf_output[tile * TILE_LEN + i];
      if (got != expected) {
        std::cout << "MISMATCH at tile=" << tile << " elem=" << i << ": got "
                  << got << " expected " << expected << "\n";
        pass = false;
      }
    }
  }

  if (pass)
    std::cout << "PASS!\n";
  else
    std::cout << "FAIL.\n";

  return pass ? 0 : 1;
}
