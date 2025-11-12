//===- AIEInlineTraceConfig.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Pass to inline trace.start_config and generate npu.write32
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

struct AIEInlineTraceConfigPass
    : AIEXInlineTraceConfigBase<AIEInlineTraceConfigPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Load RegisterDatabase for AIE2
    auto regDb = RegisterDatabase::loadAIE2();
    if (!regDb) {
      device.emitError("Failed to load register database");
      return signalPassFailure();
    }

    // Collect all trace.start_config operations
    SmallVector<TraceStartConfigOp> startConfigs;
    device.walk([&](TraceStartConfigOp startConfig) {
      startConfigs.push_back(startConfig);
    });

    for (auto startConfig : startConfigs) {
      OpBuilder builder(startConfig);

      // Lookup the trace config symbol
      auto configSymbolName = startConfig.getTraceConfig();
      auto configOp =
          dyn_cast_or_null<TraceConfigOp>(SymbolTable::lookupNearestSymbolFrom(
              device, builder.getStringAttr(configSymbolName)));

      if (!configOp) {
        startConfig.emitError("trace config symbol '")
            << configSymbolName << "' not found";
        return signalPassFailure();
      }

      // Get tile and extract col/row
      auto tile = configOp.getTile();
      auto tileOp = dyn_cast<TileOp>(tile.getDefiningOp());
      if (!tileOp) {
        startConfig.emitError("tile operand must be a TileOp");
        return signalPassFailure();
      }

      int col = tileOp.getCol();
      int row = tileOp.getRow();

      // Process all trace.reg operations in the config
      for (auto &op : configOp.getBody().getOps()) {
        auto regOp = dyn_cast<TraceRegOp>(op);
        if (!regOp)
          continue;

        // After packing, field should not be present
        if (regOp.getField()) {
          regOp.emitError("aie.trace.reg still has field attribute - run "
                          "-aie-trace-pack-reg-writes pass first");
          return signalPassFailure();
        }

        // Look up register to get offset
        auto regName = regOp.getRegName().str();
        const RegisterInfo *regInfo =
            regDb->lookupRegister(regName, tileOp, /*isMem=*/false);
        if (!regInfo) {
          regOp.emitError("Register '") << regName << "' not found for tile ("
                                        << col << ", " << row << ")";
          return signalPassFailure();
        }

        // Extract value (mask is discarded)
        uint32_t value = 0;
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(regOp.getValue())) {
          value = intAttr.getInt();
        } else {
          regOp.emitError("value must be an integer after packing");
          return signalPassFailure();
        }

        // Generate aiex.npu.write32 operation with col/row
        builder.create<AIEX::NpuWrite32Op>(
            regOp.getLoc(), builder.getUI32IntegerAttr(regInfo->offset),
            builder.getUI32IntegerAttr(value),
            nullptr,                        // buffer
            builder.getI32IntegerAttr(col), // column
            builder.getI32IntegerAttr(row)  // row
        );
      }

      // Remove the start_config invocation
      startConfig.erase();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEXInlineTraceConfigPass() {
  return std::make_unique<AIEInlineTraceConfigPass>();
}
