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
#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEConfigToNPUPass : AIEConfigToNPUBase<AIEConfigToNPUPass> {
  void runOnOperation() override {
    // This pass is now a no-op.
    // NPU write generation happens in AIEInlineTraceConfig (Pass 2).
    // Keeping this pass for future extensibility.
    return;
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEConfigToNPUPass() {
  return std::make_unique<AIEConfigToNPUPass>();
}
