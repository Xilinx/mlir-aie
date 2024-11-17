//===- target_model_rtti.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <stdexcept>

using namespace xilinx;

void test() {

  // AIEDevice::xcvc1902
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).isNPU()) {
    throw std::runtime_error("Failed xcvc1902 isNPU returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).isUsingSemaphoreLocks()) {
    throw std::runtime_error(
        "Failed xcvc1902 isUsingSemaphoreLocks returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).columns() != 50) {
    throw std::runtime_error("Failed xcvc1902 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).rows() != 9) {
    throw std::runtime_error("Failed xcvc1902 rows");
  }

  // AIEDevice::xcve2302
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).isNPU()) {
    throw std::runtime_error("Failed xcve2302 isNPU returns true");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2302).isUsingSemaphoreLocks()) {
    throw std::runtime_error("Failed xcve2302 isUsingSemaphoreLocks");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).columns() != 17) {
    throw std::runtime_error("Failed xcve2302 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).rows() != 4) {
    throw std::runtime_error("Failed xcve2302 rows");
  }

  // AIEDevice::xcve2802
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).isNPU()) {
    throw std::runtime_error("Failed xcve2802 isNPU returns true");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2802).isUsingSemaphoreLocks()) {
    throw std::runtime_error("Failed xcve2802 isUsingSemaphoreLocks");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).columns() != 38) {
    throw std::runtime_error("Failed xcve2802 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).rows() != 11) {
    throw std::runtime_error("Failed xcve2802 rows");
  }

  // AIEDevice::npu1
  if (!AIE::getTargetModel(AIE::AIEDevice::npu1).isNPU()) {
    throw std::runtime_error("Failed npu1 isNPU");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu1).isUsingSemaphoreLocks()) {
    throw std::runtime_error("Failed npu1 isUsingSemaphoreLocks");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu1).columns() != 5) {
    throw std::runtime_error("Failed npu1 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu1).rows() != 6) {
    throw std::runtime_error("Failed npu1 rows");
  }

  // AIEDevice::npu_1col, npu_2col, npu_3col, npu_4col
  llvm::DenseMap<AIE::AIEDevice, int> npu1_devs;
  npu1_devs[AIE::AIEDevice::npu1_1col] = 1;
  npu1_devs[AIE::AIEDevice::npu1_2col] = 2;
  npu1_devs[AIE::AIEDevice::npu1_3col] = 3;
  npu1_devs[AIE::AIEDevice::npu1_4col] = 4;
  for (auto &[dev, cols] : npu1_devs) {
    if (!AIE::getTargetModel(dev).isNPU()) {
      throw std::runtime_error("Failed npu1_ncol isNPU");
    }
    if (!AIE::getTargetModel(dev).isUsingSemaphoreLocks()) {
      throw std::runtime_error("Failed npu1_ncol isUsingSemaphoreLocks");
    }
    if (AIE::getTargetModel(dev).columns() != cols) {
      throw std::runtime_error("Failed npu1_ncol columns");
    }
    if (AIE::getTargetModel(dev).rows() != 6) {
      throw std::runtime_error("Failed npu1_ncol rows");
    }
  }

  // AIEDevice::npu2
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2).isNPU()) {
    throw std::runtime_error("Failed npu2 isNPU");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2).isUsingSemaphoreLocks()) {
    throw std::runtime_error("Failed npu2 isUsingSemaphoreLocks");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2).columns() != 8) {
    throw std::runtime_error("Failed npu2 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2).rows() != 6) {
    throw std::runtime_error("Failed npu2 rows");
  }
}

int main() {
  test();
  return 0;
}