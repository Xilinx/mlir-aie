//===- AIEConfigToNPU.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Pass to lower trace.reg to aiex.npu.write32
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEConfigToNPUPass : AIEConfigToNPUBase<AIEConfigToNPUPass> {
  void runOnOperation() override {
    // Stub implementation for now
    // Will be implemented in next commit
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEConfigToNPUPass() {
  return std::make_unique<AIEConfigToNPUPass>();
}
