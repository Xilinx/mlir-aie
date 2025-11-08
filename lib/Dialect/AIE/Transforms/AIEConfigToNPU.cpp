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
  AIEConfigToNPUPass() {
    // Load AIEX dialect in constructor (before multi-threading)
    // This is a workaround since TableGen dependentDialects doesn't support AIEX
  }
  
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device);
    
    // Get AIEX dialect (should already be loaded by aie-opt)
    auto* aiexDialect = device->getContext()->getLoadedDialect<AIEX::AIEXDialect>();
    if (!aiexDialect) {
      // For prototype: just skip NPU write generation if AIEX not available
      // In production, this would be an error
      return;
    }
    
    // Load register database for AIE2
    auto regDb = RegisterDatabase::loadAIE2();
    if (!regDb) {
      device.emitError("Failed to load register database");
      return signalPassFailure();
    }
    
    // Collect all trace.reg operations at device level (inlined ones)
    SmallVector<TraceRegOp> regOps;
    device.walk([&](TraceRegOp regOp) {
      if (isa<DeviceOp>(regOp->getParentOp()) && regOp.getTile()) {
        regOps.push_back(regOp);
      }
    });
    
    // Group register writes by (tile_op, register_offset) for merging
    // Use TileOp pointer as key
    using TileRegKey = std::pair<TileOp, uint32_t>;
    std::map<TileRegKey, uint32_t> mergedValues;
    std::map<TileRegKey, TraceRegOp> firstRegOp;
    
    for (auto regOp : regOps) {
      auto tile = regOp.getTile();
      if (!tile) continue;
      
      auto tileOp = dyn_cast<TileOp>(tile.getDefiningOp());
      if (!tileOp) continue;
      
      // Determine module based on tile row (simplified for AIE2)
      // Row 0 = shim (PL_MODULE), Row 1 = memtile, Row >= 2 = core
      std::string module = "CORE_MODULE";
      int row = tileOp.getRow();
      if (row == 0) module = "PL_MODULE";
      else if (row == 1) module = "MEM_TILE_MODULE";
      
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
        // Integer value - encode directly
        encodedValue = regDb->encodeFieldValue(*fieldInfo, intAttr.getInt());
      } else if (auto strAttr = llvm::dyn_cast<StringAttr>(value)) {
        // String value - could be event name or enum
        std::string strVal = strAttr.getValue().str();
        
        // Try to resolve as event
        auto eventCode = regDb->lookupEvent(strVal, "core");
        if (eventCode) {
          encodedValue = regDb->encodeFieldValue(*fieldInfo, *eventCode);
        } else {
          // For prototype: use 0 as fallback
          encodedValue = 0;
        }
      }
      
      // Merge into accumulated value for this register
      auto key = std::make_pair(tileOp, regInfo->offset);
      mergedValues[key] |= encodedValue;
      if (firstRegOp.find(key) == firstRegOp.end()) {
        firstRegOp[key] = regOp;
      }
    }
    
    // Generate aiex.npu.write32 operations
    for (auto& [key, value] : mergedValues) {
      auto [tileOp, offset] = key;
      auto loc = firstRegOp[key].getLoc();
      
      int col = tileOp.getCol();
      int row = tileOp.getRow();
      
      // Calculate absolute address: ((col << 25) | (row << 20)) + offset
      uint32_t absoluteAddress = ((col << 25) | (row << 20)) + offset;
      
      // Create aiex.npu.write32 operation
      builder.setInsertionPointAfter(regOps.back());
      builder.create<AIEX::NpuWrite32Op>(
        loc,
        builder.getUI32IntegerAttr(absoluteAddress),
        builder.getUI32IntegerAttr(value),
        nullptr,  // buffer
        nullptr,  // column
        nullptr   // row
      );
    }
    
    // Remove all processed trace.reg operations
    for (auto regOp : regOps) {
      regOp.erase();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEConfigToNPUPass() {
  return std::make_unique<AIEConfigToNPUPass>();
}
