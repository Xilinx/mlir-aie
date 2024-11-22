//===- target_model.cpp -----------------------------------------*- C++ -*-===//
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
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'UsesSemaphoreLocks' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcvc1902 property check for "
                             "'UsesMultiDimensionalBDs' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).columns() != 50) {
    throw std::runtime_error("Failed xcvc1902 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).rows() != 9) {
    throw std::runtime_error("Failed xcvc1902 rows");
  }

  // AIEDevice::xcve2302
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2302)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2302)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcve2302 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).columns() != 17) {
    throw std::runtime_error("Failed xcve2302 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).rows() != 4) {
    throw std::runtime_error("Failed xcve2302 rows");
  }

  // AIEDevice::xcve2802
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2802)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2802)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcve2802 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).columns() != 38) {
    throw std::runtime_error("Failed xcve2802 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).rows() != 11) {
    throw std::runtime_error("Failed xcve2802 rows");
  }

  // AIEDevice::npu1
  if (!AIE::getTargetModel(AIE::AIEDevice::npu1)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error(
        "Failed npu1 property check for 'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu1)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed npu1 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu1)
           .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed npu1 property check for 'IsNPU' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu1)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed npu1 property check for 'IsVirtualized' returns true");
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
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesSemaphoreLocks)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'UsesSemaphoreLocks' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'UsesMultiDimensionalBDs' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(AIE::AIETargetModel::IsNPU)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'IsNPU' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::IsVirtualized)) {
      throw std::runtime_error(
          "Failed npu1_ncol property check for 'IsVirtualized' returns false");
    }
    if (AIE::getTargetModel(dev).columns() != cols) {
      throw std::runtime_error("Failed npu1_ncol columns");
    }
    if (AIE::getTargetModel(dev).rows() != 6) {
      throw std::runtime_error("Failed npu1_ncol rows");
    }
  }

  // AIEDevice::npu2
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed npu2 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'IsNPU' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'IsVirtualized' returns true");
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
