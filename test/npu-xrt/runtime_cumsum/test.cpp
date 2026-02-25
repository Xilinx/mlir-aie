//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

using INOUT_DATATYPE = int32_t;

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v;

  instr_v = test_utils::load_instr_binary("insts.bin");

  int INOUT_SIZE = 128;

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  std::string xclbin_name = "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);

  std::string Node = "MLIR_AIE";

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, INOUT_SIZE * sizeof(INOUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  INOUT_DATATYPE *bufInA = bo_inA.map<INOUT_DATATYPE *>();
  std::vector<INOUT_DATATYPE> srcVecA(INOUT_SIZE);
  for (int j = 0; j < 1; j++) {
    for (int i = 0; i < 16; i++) {
      srcVecA[j * 16 + i] = 1;
    }
  }
  printf("Input:\n");
  for (int i = 0; i < INOUT_SIZE; i++) {
    printf("%d\t", srcVecA[i]);
    if ((i + 1) % 16 == 0) {
      printf("\n");
    }
  }
  printf("\n");

  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(INOUT_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_inA.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  INOUT_DATATYPE *bufOut = bo_inA.map<INOUT_DATATYPE *>();

  std::vector<INOUT_DATATYPE> ref(INOUT_SIZE);
  for (int i = 0; i < 16; i++) {
    ref[i] = srcVecA[i];
    ref[1 * 16 + i] = srcVecA[i];
    ref[2 * 16 + i] = ref[1 * 16 + i] + ref[i];
    ref[3 * 16 + i] = ref[2 * 16 + i] + ref[1 * 16 + i] + ref[i];
    ref[4 * 16 + i] =
        ref[3 * 16 + i] + ref[2 * 16 + i] + ref[1 * 16 + i] + ref[i];
    ref[5 * 16 + i] = ref[4 * 16 + i] + ref[3 * 16 + i] + ref[2 * 16 + i] +
                      ref[1 * 16 + i] + ref[i];
    ref[6 * 16 + i] = ref[5 * 16 + i] + ref[4 * 16 + i] + ref[3 * 16 + i] +
                      ref[2 * 16 + i] + ref[1 * 16 + i] + ref[i];
    ref[7 * 16 + i] = ref[6 * 16 + i] + ref[5 * 16 + i] + ref[4 * 16 + i] +
                      ref[3 * 16 + i] + ref[2 * 16 + i] + ref[1 * 16 + i] +
                      ref[i];
  }

  int errors = 0;

  printf("Output:\n");
  for (uint32_t i = 0; i < INOUT_SIZE; i++) {
    printf("%d\t", bufOut[i]);
    if ((i + 1) % 16 == 0) {
      printf("\n");
    }
  }
  printf("\n");

  for (uint32_t i = 0; i < INOUT_SIZE; i++) {
    if (*(bufOut + i) != ref[i]) {
      if (errors < 10) {
        std::cout << "Error in output " << i << "; Input: " << srcVecA[i]
                  << "; Output: " << *(bufOut + i) << " != reference:" << ref[i]
                  << std::endl;
      }
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
