//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int inSize = 64;
constexpr int outSize = 64;

std::vector<uint32_t> loadInstrSequence(const std::string &instrPath) {
  std::ifstream instrFile(instrPath);
  std::string line;
  std::vector<uint32_t> instrV;
  while (std::getline(instrFile, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instrV.push_back(a);
  }
  return instrV;
}

int main(int argc, const char *argv[]) {
  auto xclbin = xrt::xclbin("final.xclbin");
  std::string node = "MLIR_AIE";
  std::vector<uint32_t> instrV = loadInstrSequence("insts.txt");

  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);

  xrt::hw_context context(device, xclbin.get_uuid());

  auto kernel = xrt::kernel(context, kernelName);

  auto boInstr = xrt::bo(device, instrV.size() * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto boInA = xrt::bo(device, inSize * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(2));
  auto boInB = xrt::bo(device, inSize * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto boOut = xrt::bo(device, outSize * sizeof(int32_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  uint32_t *bufInA = boInA.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < inSize; i++)
    srcVecA.push_back(i + 1);
  std::memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  void *bufInstr = boInstr.map<void *>();
  std::memcpy(bufInstr, instrV.data(), instrV.size() * sizeof(int));

  boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  boInA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(boInstr, instrV.size(), boInA, boInB, boOut);
  run.wait();

  boOut.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = boOut.map<uint32_t *>();

  int errors = 0;

  for (uint32_t i = 0; i < 64; i++) {
    uint32_t ref = i + 2;
    if (*(bufOut + i) != ref) {
      std::cout << "Error in output " << *(bufOut + i) << " != " << ref
                << std::endl;
      errors++;
    } else {
      std::cout << "Correct output " << *(bufOut + i) << " == " << ref
                << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nfailed.\n\n";
  return 1;
}
