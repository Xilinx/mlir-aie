//===- AIEExpandLoadPdi.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass expands `npu.load_pdi` operations that reference a device into:
// 1. An empty device PDI load (causes firmware to reset the device)
// 2. Optional reset operations based on command-line configuration
// 3. Explicit configuration operations (write32/blockwrite)
//
// Example command-line usage:
//   aie-opt --aie-expand-load-pdi \
//     --reset-switches-tiles=shim,mem,core \
//     --reset-switches-mode=ifused \
//     --reset-locks-tiles=mem,core \
//     --reset-locks-mode=always \
//     input.mlir
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aie-expand-load-pdi"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;
using namespace xilinx::AIE;

namespace {

// Helper function to parse tile types from comma-separated string
static ResetTileType parseTileTypes(llvm::StringRef str) {
  if (str.empty())
    return ResetTileType::None;
    
  unsigned result = 0;
  llvm::SmallVector<llvm::StringRef> tokens;
  str.split(tokens, ',', -1, false);
  
  for (auto token : tokens) {
    token = token.trim();
    if (token == "shim" || token == "shimnoc")
      result |= static_cast<unsigned>(ResetTileType::ShimNOC);
    else if (token == "mem" || token == "memtile")
      result |= static_cast<unsigned>(ResetTileType::MemTile);
    else if (token == "core" || token == "coretile")
      result |= static_cast<unsigned>(ResetTileType::CoreTile);
    else if (token == "all")
      return ResetTileType::All;
  }
  
  return static_cast<ResetTileType>(result);
}

// Helper function to parse reset mode from string
static ResetMode parseResetMode(llvm::StringRef str) {
  auto trimmed = str.trim().lower();
  if (trimmed == "never")
    return ResetMode::Never;
  else if (trimmed == "ifused" || trimmed == "if-used")
    return ResetMode::IfUsed;
  else if (trimmed == "ifusedfinegrained" || trimmed == "if-used-fine-grained")
    return ResetMode::IfUsedFineGrained;
  else if (trimmed == "ifchanged" || trimmed == "if-changed")
    return ResetMode::IfChanged;
  else if (trimmed == "ifchangedfinegrained" || trimmed == "if-changed-fine-grained")
    return ResetMode::IfChangedFineGrained;
  else if (trimmed == "always" || trimmed == "all")
    return ResetMode::Always;
  else
    return ResetMode::Never;
}

struct ExpandLoadPdiPattern : public OpRewritePattern<NpuLoadPdiOp> {
  const ResetConfig dmaConfig;
  const ResetConfig switchConfig;
  const ResetConfig lockConfig;
  const ResetConfig coreConfig;
  mutable AIE::DeviceOp previousDevice;
  
  ExpandLoadPdiPattern(MLIRContext *context, ResetConfig dma, ResetConfig sw, ResetConfig lock, ResetConfig core)
      : OpRewritePattern<NpuLoadPdiOp>(context), 
        dmaConfig(dma), switchConfig(sw), lockConfig(lock), coreConfig(core), previousDevice(nullptr) {}

  LogicalResult matchAndRewrite(NpuLoadPdiOp loadPdiOp,
                                PatternRewriter &rewriter) const override {
    // Only process load_pdi ops that reference a device
    auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
    if (!deviceRefAttr)
      return failure();

    // Look up the referenced device
    auto moduleOp = loadPdiOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
    if (!referencedDevice) {
      loadPdiOp.emitError("Referenced symbol '")
          << deviceRefAttr.getValue() << "' is not a device";
      return failure();
    }

    // If already referencing an empty device, don't transform
    if (referencedDevice.getSymName().starts_with("empty"))
      return failure();

    // Determine if this is the first load_pdi (no previous device)
    bool isFirstLoadPdi = !previousDevice;
    
    // For first load_pdi with no reset modes requiring comparison, skip empty device
    bool needsEmptyDevice = !isFirstLoadPdi || 
                           dmaConfig.mode != ResetMode::Never || 
                           switchConfig.mode != ResetMode::Never || 
                           lockConfig.mode != ResetMode::Never;

    AIE::DeviceOp emptyDevice;
    if (needsEmptyDevice) {
      // Create a unique empty device for this reset to avoid PDI address caching
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      // Find a unique name for the empty device
      std::string emptyName;
      int emptyIndex = 0;
      do {
        emptyName = "empty_" + std::to_string(emptyIndex++);
      } while (moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName));
      
      auto deviceType = referencedDevice.getDevice();
      auto loc = rewriter.getUnknownLoc();
      emptyDevice = AIE::DeviceOp::create(
        rewriter, loc, deviceType, rewriter.getStringAttr(emptyName));
      emptyDevice.getRegion().emplaceBlock();
      
      Block *deviceBlock = &emptyDevice.getRegion().front();
      rewriter.setInsertionPointToEnd(deviceBlock);
      AIE::EndOp::create(rewriter, loc);
    }

    // Create new load_pdi operation
    rewriter.setInsertionPoint(loadPdiOp);
    NpuLoadPdiOp newLoadPdi;
    if (needsEmptyDevice) {
      newLoadPdi = NpuLoadPdiOp::create(
          rewriter, loadPdiOp.getLoc(), FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
          loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
          loadPdiOp.getAddressAttr());
    } else {
      // First load_pdi with no resets - keep original device reference
      newLoadPdi = NpuLoadPdiOp::create(
          rewriter, loadPdiOp.getLoc(), loadPdiOp.getDeviceRefAttr(),
          loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
          loadPdiOp.getAddressAttr());
    }

    // Generate and insert reset operations (skip for first load_pdi)
    if (!isFirstLoadPdi && 
        (dmaConfig.mode != ResetMode::Never || 
         switchConfig.mode != ResetMode::Never || 
         lockConfig.mode != ResetMode::Never ||
         coreConfig.mode != ResetMode::Never)) {
      if (failed(xilinx::AIE::generateAndInsertResetOps(
              referencedDevice, newLoadPdi.getOperation(),
              dmaConfig, switchConfig, lockConfig, coreConfig, previousDevice))) {
        loadPdiOp.emitError("Failed to generate reset operations");
        return failure();
      }
    }

    // Generate and insert the configuration operations after the reset ops
    // Find the last operation inserted (which should be from reset ops)
    Operation *lastResetOp = newLoadPdi.getOperation();
    Operation *nextOp = newLoadPdi->getNextNode();
    while (nextOp && nextOp != loadPdiOp.getOperation()) {
      lastResetOp = nextOp;
      nextOp = nextOp->getNextNode();
    }

    if (failed(xilinx::AIE::generateAndInsertConfigOps(
            referencedDevice, lastResetOp, ""))) {
      loadPdiOp.emitError("Failed to generate configuration operations");
      return failure();
    }

    // Remove the original load_pdi operation
    rewriter.eraseOp(loadPdiOp);

    // Update previous device for next load_pdi
    previousDevice = referencedDevice;

    return success();
  }
};

struct AIEExpandLoadPdiPass
    : public AIEExpandLoadPdiBase<AIEExpandLoadPdiPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Parse reset configurations from command-line options
    ResetTileType dmaTiles = parseTileTypes(resetDmasTiles);
    ResetMode dmaMode = parseResetMode(resetDmasMode);
    ResetConfig dmaConfig(dmaTiles, dmaMode);
    
    ResetTileType switchTiles = parseTileTypes(resetSwitchesTiles);
    ResetMode switchMode = parseResetMode(resetSwitchesMode);
    ResetConfig switchConfig(switchTiles, switchMode);
    
    ResetTileType lockTiles = parseTileTypes(resetLocksTiles);
    ResetMode lockMode = parseResetMode(resetLocksMode);
    ResetConfig lockConfig(lockTiles, lockMode);
    
    ResetTileType coreTiles = parseTileTypes(resetCoresTiles);
    ResetMode coreMode = parseResetMode(resetCoresMode);
    ResetConfig coreConfig(coreTiles, coreMode);

    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandLoadPdiPattern>(&getContext(), dmaConfig, switchConfig, lockConfig, coreConfig);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
