//===- AIEInlineTraceConfig.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Pass to inline trace.start_config into runtime sequence
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEInlineTraceConfigPass : AIEInlineTraceConfigBase<AIEInlineTraceConfigPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    
    // Collect all trace.start_config operations
    SmallVector<TraceStartConfigOp> startConfigs;
    device.walk([&](TraceStartConfigOp startConfig) {
      startConfigs.push_back(startConfig);
    });
    
    for (auto startConfig : startConfigs) {
      OpBuilder builder(startConfig);
      
      // Lookup the trace config symbol
      auto configSymbolName = startConfig.getTraceConfig();
      auto configOp = dyn_cast_or_null<TraceConfigOp>(
        SymbolTable::lookupNearestSymbolFrom(device, 
          builder.getStringAttr(configSymbolName))
      );
      
      if (!configOp) {
        startConfig.emitError("trace config symbol '")
          << configSymbolName << "' not found";
        return signalPassFailure();
      }
      
      // Clone all register write ops from config to call site
      for (auto &op : configOp.getBody().getOps()) {
        if (auto regOp = dyn_cast<TraceRegOp>(op)) {
          // Create new reg op at call site
          builder.create<TraceRegOp>(
            regOp.getLoc(),
            regOp.getRegNameAttr(),
            regOp.getFieldAttr(),
            regOp.getValueAttr(),
            regOp.getCommentAttr()
          );
        }
      }
      
      // Remove the start_config invocation
      startConfig.erase();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEInlineTraceConfigPass() {
  return std::make_unique<AIEInlineTraceConfigPass>();
}
