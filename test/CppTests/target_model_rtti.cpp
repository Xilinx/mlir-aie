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
  if (!llvm::isa<AIE::VC1902TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcvc1902))) {
    throw std::runtime_error("Failed xcvc1902 isa<VC1902TargetModel>");
  }
  if (!llvm::isa<AIE::AIE1TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcvc1902))) {
    throw std::runtime_error("Failed xcvc1902 isa<AIE1TargetModel>");
  }
  if (llvm::isa<AIE::AIE2TargetModel, AIE::VE2302TargetModel,
                AIE::VE2802TargetModel, AIE::BaseNPUTargetModel,
                AIE::NPUTargetModel, AIE::VirtualizedNPUTargetModel,
                AIE::NPU2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcvc1902))) {
    throw std::runtime_error("Failed xcvc1902 !isa<>");
  }

  // AIEDevice::xcve2302
  if (!llvm::isa<AIE::VE2302TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2302))) {
    throw std::runtime_error("Failed xcve2302 isa<VE2302TargetModel>");
  }
  if (!llvm::isa<AIE::AIE2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2302))) {
    throw std::runtime_error("Failed xcve2302 isa<AIE2TargetModel>");
  }
  if (llvm::isa<AIE::AIE1TargetModel, AIE::VC1902TargetModel,
                AIE::VE2802TargetModel, AIE::BaseNPUTargetModel,
                AIE::NPUTargetModel, AIE::VirtualizedNPUTargetModel,
                AIE::NPU2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2302))) {
    throw std::runtime_error("Failed xcve2302 !isa<>");
  }

  // AIEDevice::xcve2802
  if (!llvm::isa<AIE::VE2802TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2802))) {
    throw std::runtime_error("Failed xcvc1902 isa<VE2802TargetModel>");
  }
  if (!llvm::isa<AIE::AIE2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2802))) {
    throw std::runtime_error("Failed xcve2802 isa<AIE2TargetModel>");
  }
  if (llvm::isa<AIE::AIE1TargetModel, AIE::VC1902TargetModel,
                AIE::VE2302TargetModel, AIE::BaseNPUTargetModel,
                AIE::NPUTargetModel, AIE::VirtualizedNPUTargetModel,
                AIE::NPU2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::xcve2802))) {
    throw std::runtime_error("Failed xcve2802 !isa<>");
  }

  // AIEDevice::npu1
  if (!llvm::isa<AIE::AIE2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu1))) {
    throw std::runtime_error("Failed npu1 isa<AIE2TargetModel>");
  }
  if (!llvm::isa<AIE::BaseNPUTargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu1))) {
    throw std::runtime_error("Failed npu1 isa<BaseNPUTargetModel>");
  }
  if (!llvm::isa<AIE::NPUTargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu1))) {
    throw std::runtime_error("Failed npu1 isa<NPUTargetModel>");
  }
  if (llvm::isa<AIE::AIE1TargetModel, AIE::VC1902TargetModel,
                AIE::VE2302TargetModel, AIE::VE2802TargetModel,
                AIE::VirtualizedNPUTargetModel, AIE::NPU2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu1))) {
    throw std::runtime_error("Failed npu1 !isa<>");
  }

  // AIEDevice::npu_1col, npu_2col, npu_3col, npu_4col
  llvm::SmallVector<AIE::AIEDevice> npu1_devs = {
      AIE::AIEDevice::npu1_1col, AIE::AIEDevice::npu1_2col,
      AIE::AIEDevice::npu1_3col, AIE::AIEDevice::npu1_4col};
  for (auto dev : npu1_devs) {
    if (!llvm::isa<AIE::AIE2TargetModel>(AIE::getTargetModel(dev))) {
      throw std::runtime_error("Failed npu1_col isa<AIE2TargetModel>");
    }
    if (!llvm::isa<AIE::VirtualizedNPUTargetModel>(AIE::getTargetModel(dev))) {
      throw std::runtime_error(
          "Failed npu1_col isa<VirtualizedNPUTargetModel>");
    }
    if (!llvm::isa<AIE::BaseNPUTargetModel>(AIE::getTargetModel(dev))) {
      throw std::runtime_error("Failed npu1_col isa<BaseNPUTargetModel>");
    }
    if (llvm::isa<AIE::AIE1TargetModel, AIE::VC1902TargetModel,
                  AIE::VE2302TargetModel, AIE::VE2802TargetModel,
                  AIE::NPUTargetModel, AIE::NPU2TargetModel>(
            AIE::getTargetModel(dev))) {
      throw std::runtime_error("Failed npu1_col !isa<>");
    }
  }

  // AIEDevice::npu2
  if (!llvm::isa<AIE::AIE2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu2))) {
    throw std::runtime_error("Failed npu2 isa<AIE2TargetModel>");
  }
  if (!llvm::isa<AIE::BaseNPUTargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu2))) {
    throw std::runtime_error("Failed npu2 isa<BaseNPUTargetModel>");
  }
  if (!llvm::isa<AIE::NPU2TargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu2))) {
    throw std::runtime_error("Failed npu2 isa<NPU2TargetModel>");
  }
  if (llvm::isa<AIE::AIE1TargetModel, AIE::VC1902TargetModel,
                AIE::VE2302TargetModel, AIE::VE2802TargetModel,
                AIE::VirtualizedNPUTargetModel, AIE::NPUTargetModel>(
          AIE::getTargetModel(AIE::AIEDevice::npu2))) {
    throw std::runtime_error("Failed npu2 !isa<>");
  }
}

int main() {
  test();
  return 0;
}