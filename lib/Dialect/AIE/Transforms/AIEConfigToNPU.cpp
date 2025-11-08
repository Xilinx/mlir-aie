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
    DeviceOp device = getOperation();
    OpBuilder builder(device);
    
    // For prototype: simplified implementation
    // We'll generate placeholder npu.write32 operations
    // A full implementation would:
    // 1. Load RegisterDatabase
    // 2. Resolve register names to offsets
    // 3. Merge bitfields for same register
    // 4. Calculate absolute addresses
    
    // For now, just emit a comment that this pass ran
    // The actual implementation will be added when we integrate
    // with a real runtime sequence and AIEX dialect
    
    // Collect all trace.reg operations at device level
    SmallVector<TraceRegOp> regOps;
    device.walk([&](TraceRegOp regOp) {
      // Only process reg ops that are direct children of device
      // (i.e., inlined ones, not ones still in trace.config)
      if (isa<DeviceOp>(regOp->getParentOp())) {
        regOps.push_back(regOp);
      }
    });
    
    // For prototype: just verify we can iterate them
    // Full implementation would generate aiex.npu.write32 here
    (void)regOps;  // Suppress unused variable warning
    
    // Placeholder: In full implementation, this would:
    // 1. Look up register offset from database
    // 2. Encode field value
    // 3. Merge with other writes to same register
    // 4. Generate aiex.npu.write32
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEConfigToNPUPass() {
  return std::make_unique<AIEConfigToNPUPass>();
}
