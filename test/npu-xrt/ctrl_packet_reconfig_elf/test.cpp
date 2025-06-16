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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_module.h"

#include "test_utils.h"

constexpr int IN_SIZE = 64 * 64;
constexpr int OUT_SIZE = 64 * 64;

#define IN_DATATYPE int8_t
#define OUT_DATATYPE int8_t

int main(int argc, const char *argv[]) {
  // AIE configuration control packets' raw data
  std::vector<uint32_t> ctrlPackets =
      test_utils::load_instr_binary("ctrlpkt.bin");

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  // Skeleton xclbin containing only the control packet network
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

  xrt::elf elf0("ctrlpkt_dma_seq.elf");
  xrt::module mod0{elf0};

  xrt::elf elf1("aie2_run_seq.elf");
  xrt::module mod1{elf1};

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel0 = xrt::ext::kernel(context, mod0, kernelName);
  auto kernel1 = xrt::ext::kernel(context, mod1, kernelName);

  xrt::bo bo_pkt = xrt::ext::bo(device, ctrlPackets.size() * sizeof(int32_t));
  xrt::bo bo_inA = xrt::ext::bo(device, IN_SIZE * sizeof(IN_DATATYPE));
  xrt::bo bo_out = xrt::ext::bo(device, OUT_SIZE * sizeof(OUT_DATATYPE));

  IN_DATATYPE *bufInA = bo_inA.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back((i % 64) + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(IN_DATATYPE)));

  void *bufpkt = bo_pkt.map<void *>();
  memcpy(bufpkt, ctrlPackets.data(), ctrlPackets.size() * sizeof(int));

  // Synchronizing BOs
  bo_pkt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  kernel0(opcode, 0, 0, bo_pkt).wait2();
  kernel1(opcode, 0, 0, bo_inA, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = srcVecA[i] + 12;
    if (bufOut[i] != ref) {
      std::cout << "Error in output " << std::to_string(bufOut[i])
                << " != " << ref << std::endl;
      errors++;
    }
    // else
    //   std::cout << "Correct output " << std::to_string(bufOut[i])
    //             << " == " << ref << std::endl;
    // }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }

  std::cout << "\nfailed.\n\n";
  return 1;
}
