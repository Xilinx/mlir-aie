//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

constexpr int IN_SIZE = 4 * 64 * 64;
constexpr int OUT_SIZE = 4 * 64 * 64;

#define IN_DATATYPE int8_t
#define OUT_DATATYPE int8_t

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary("aie2_run_seq.bin");
  std::vector<uint32_t> ctrlPackets =
      test_utils::load_instr_binary("ctrlpkt.bin");

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  auto xclbin = xrt::xclbin(std::string("aie1.xclbin"));

  std::string Node = "MLIR_AIE";

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_ctrlpkt = xrt::bo(device, ctrlPackets.size() * sizeof(int32_t),
                            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(IN_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(OUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  IN_DATATYPE *bufInA = bo_inA.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back((i % 128) + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(IN_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  void *bufctrlpkt = bo_ctrlpkt.map<void *>();
  memcpy(bufctrlpkt, ctrlPackets.data(), ctrlPackets.size() * sizeof(int));

  // bo_ctrlpkt_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_ctrlpkt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  // Creating a runlist to contain two seperate runs
  xrt::runlist runlist = xrt::runlist(context);

  // Run 1: the design
  auto run1 = xrt::run(kernel);
  run1.set_arg(0, opcode);
  run1.set_arg(1, bo_instr);
  run1.set_arg(2, instr_v.size());
  run1.set_arg(3, bo_inA);
  run1.set_arg(4, bo_out);
  run1.set_arg(5, bo_ctrlpkt);

  // Executing and waiting on the runlist
  runlist.add(run1);
  runlist.execute();
  runlist.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();

  int errors = 0;

  for (uint32_t core = 0; core < 4; core++) {
    for (uint32_t i = 0; i < 4096; i++) {
      OUT_DATATYPE ref = srcVecA[core * 4096 + i] + 12 + core;
      if (*(bufOut + core * 4096 + i) != ref) {
        std::cout << "Error at i=" << i << " core=" << core
                  << " output: " << std::to_string(bufOut[core * 4096 + i])
                  << " != " << (int)ref << std::endl;
        errors++;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }

  std::cout << "\nfailed.\n\n";
  std::cout << "failed count: " << errors << std::endl;
  return 1;
}
