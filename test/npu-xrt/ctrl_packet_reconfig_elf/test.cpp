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

constexpr int DATA_SIZE = 64 * 64;
#define DATATYPE int8_t

int main(int argc, const char *argv[]) {

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  // Skeleton xclbin containing only the control packet network
  auto xclbin = xrt::xclbin(std::string("aie1.xclbin"));

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind("MLIR_AIE", 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);

  xrt::elf elf0("aie2.elf");
  xrt::module mod0{elf0};

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel0 = xrt::ext::kernel(context, mod0, kernelName);

  xrt::bo bo_in = xrt::ext::bo(device, DATA_SIZE * sizeof(DATATYPE));
  xrt::bo bo_out = xrt::ext::bo(device, DATA_SIZE * sizeof(DATATYPE));

  std::vector<DATATYPE> srcVecA;
  for (int i = 0; i < DATA_SIZE; i++)
    srcVecA.push_back((i % 64) + 1);
  memcpy(bo_in.map<DATATYPE *>(), srcVecA.data(),
         (srcVecA.size() * sizeof(DATATYPE)));

  // Synchronizing BOs
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  kernel0(opcode, 0, 0, bo_in, bo_out).wait2();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  DATATYPE *bufOut = bo_out.map<DATATYPE *>();

  int errors = 0;

  for (uint32_t i = 0; i < DATA_SIZE; i++) {
    DATATYPE ref = srcVecA[i] + 3;
    if (bufOut[i] != ref) {
      std::cout << "Error in output " << std::to_string(bufOut[i])
                << " != " << (int)ref << std::endl;
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
