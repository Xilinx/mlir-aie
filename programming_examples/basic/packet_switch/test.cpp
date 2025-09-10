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

#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <cstdint>

using IN_DATATYPE = int8_t;
using OUT_DATATYPE = int8_t;

int main(int argc, const char *argv[]) {

  std::vector<uint32_t> instr_v;
  int app_id = 1;

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <app_id> <instruction_file> <bitstream_file>\n";
    return 1;
  }
  app_id = atoi(argv[1]);
  std::string instruction_file = argv[2];
  std::string bitstream_file = argv[3];

  instr_v = test_utils::load_instr_binary(instruction_file);

  int IN_SIZE = 256;
  int OUT_SIZE = 256;

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  auto xclbin = xrt::xclbin(bitstream_file);

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
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(IN_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(OUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  IN_DATATYPE *bufInA = bo_inA.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> srcVecA(IN_SIZE);
  for (int i = 0; i < IN_SIZE; i++) {
    srcVecA[i] = i % 10;
  }

  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(IN_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref;
    if (app_id == 1) {
      ref = srcVecA[i] * 2; // ref for the first input packet
    } else {
      ref = srcVecA[i] + 2; // ref for the second input packet
    }
    if (*(bufOut + i) != ref) {
      if (errors < 10) {
        std::cout << "Error in output " << i << "; Input: " << srcVecA[i]
                  << "; Output: " << *(bufOut + i) << " != reference:" << ref
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
