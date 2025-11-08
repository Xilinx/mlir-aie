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

struct AIEInlineTraceConfigPass : AIEXInlineTraceConfigBase<AIEInlineTraceConfigPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    
    // Load RegisterDatabase for AIE2
    auto regDb = RegisterDatabase::loadAIE2();
    if (!regDb) {
      device.emitError("Failed to load register database");
      return signalPassFailure();
    }
    
    // // Load AIEX dialect (it's registered in aie-opt)
    // device->getContext()->getOrLoadDialect<AIEX::AIEXDialect>();
    
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
      
      // Get tile and extract col/row
      auto tile = configOp.getTile();
      auto tileOp = dyn_cast<TileOp>(tile.getDefiningOp());
      if (!tileOp) {
        startConfig.emitError("tile operand must be a TileOp");
        return signalPassFailure();
      }
      
      int col = tileOp.getCol();
      int row = tileOp.getRow();
      
      // Determine module based on tile row
      std::string module = "CORE_MODULE";
      if (row == 0) module = "PL_MODULE";
      else if (row == 1) module = "MEM_TILE_MODULE";
      
      // Group register writes by offset for merging
      std::map<uint32_t, uint32_t> mergedValues;
      std::map<uint32_t, TraceRegOp> firstRegOp;
      
      // Process all trace.reg operations in the config
      for (auto &op : configOp.getBody().getOps()) {
        auto regOp = dyn_cast<TraceRegOp>(op);
        if (!regOp) continue;
        
        // Look up register
        auto regName = regOp.getRegName().str();
        auto* regInfo = regDb->lookupRegister(regName, module);
        if (!regInfo) {
          regOp.emitWarning("Register '") << regName << "' not found in module " << module;
          continue;
        }
        
        // Look up field
        auto fieldName = regOp.getField().str();
        auto* fieldInfo = regInfo->getField(fieldName);
        if (!fieldInfo) {
          regOp.emitWarning("Field '") << fieldName << "' not found in register " << regName;
          continue;
        }
        
        // Encode value
        uint32_t encodedValue = 0;
        auto value = regOp.getValue();
        
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
          // Integer value
          encodedValue = regDb->encodeFieldValue(*fieldInfo, intAttr.getInt());
        } else if (auto strAttr = llvm::dyn_cast<StringAttr>(value)) {
          // String value - resolve as event
          std::string strVal = strAttr.getValue().str();
          auto eventCode = regDb->lookupEvent(strVal, "core");
          if (eventCode) {
            encodedValue = regDb->encodeFieldValue(*fieldInfo, *eventCode);
          }
        }
        
        // Merge into accumulated value
        mergedValues[regInfo->offset] |= encodedValue;
        if (firstRegOp.find(regInfo->offset) == firstRegOp.end()) {
          firstRegOp[regInfo->offset] = regOp;
        }
      }
      
      // Generate aiex.npu.write32 operations with col/row
      for (auto& [offset, value] : mergedValues) {
        builder.create<AIEX::NpuWrite32Op>(
          firstRegOp[offset].getLoc(),
          builder.getUI32IntegerAttr(offset),
          builder.getUI32IntegerAttr(value),
          nullptr,  // buffer
          builder.getI32IntegerAttr(col),  // column
          builder.getI32IntegerAttr(row)   // row
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
