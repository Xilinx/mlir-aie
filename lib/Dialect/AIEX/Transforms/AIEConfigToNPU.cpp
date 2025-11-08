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
// NOTE: This pass is now a stub. NPU generation moved to AIEInlineTraceConfig.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct AIEConfigToNPUPass : AIEXConfigToNPUBase<AIEConfigToNPUPass> {
  void runOnOperation() override {
    // This pass is now a no-op.
    // NPU write generation happens in AIEInlineTraceConfig (Pass 2).
    // Keeping this pass for future extensibility.
    return;
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEXConfigToNPUPass() {
  return std::make_unique<AIEConfigToNPUPass>();
}
