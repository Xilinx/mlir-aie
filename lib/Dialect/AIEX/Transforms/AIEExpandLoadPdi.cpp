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
#include "mlir/Pass/Pass.h"

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
  else if (trimmed == "ifchangedfinegrained" ||
           trimmed == "if-changed-fine-grained")
    return ResetMode::IfChangedFineGrained;
  else if (trimmed == "always" || trimmed == "all")
    return ResetMode::Always;
  else
    return ResetMode::Never;
}

// Helper to transform a single load_pdi operation
static LogicalResult
transformLoadPdi(NpuLoadPdiOp loadPdiOp, AIE::DeviceOp previousDevice,
                 ModuleOp moduleOp, const ResetConfig &dmaConfig,
                 const ResetConfig &switchConfig, const ResetConfig &lockConfig,
                 const ResetConfig &coreConfig) {
  OpBuilder builder(loadPdiOp);

  // Only process load_pdi ops that reference a device
  auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
  if (!deviceRefAttr) {
    return success();
  }

  auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
  if (!referencedDevice) {
    loadPdiOp.emitError("Referenced symbol '")
        << deviceRefAttr.getValue() << "' is not a device";
    return failure();
  }

  AIE::DeviceOp emptyDevice;
  if (previousDevice) {
    // Create a unique empty device for this reset to avoid PDI address caching
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    // Find a unique name for the empty device
    std::string emptyName;
    int emptyIndex = 0;
    do {
      emptyName = "empty_" + std::to_string(emptyIndex++);
    } while (moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName));

    auto deviceType = referencedDevice.getDevice();
    auto loc = builder.getUnknownLoc();
    emptyDevice = AIE::DeviceOp::create(builder, loc, deviceType,
                                        builder.getStringAttr(emptyName));
    emptyDevice.getRegion().emplaceBlock();

    Block *deviceBlock = &emptyDevice.getRegion().front();
    builder.setInsertionPointToEnd(deviceBlock);
    AIE::EndOp::create(builder, loc);
  }

  builder.setInsertionPoint(loadPdiOp);

  if (previousDevice) {
    // Create new empty load_pdi operation
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(),
                         FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
                         loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
                         loadPdiOp.getAddressAttr());
    // Generate and insert reset operations if this is not the first load_pdi
    if (failed(xilinx::AIE::generateAndInsertResetOps(
            builder, referencedDevice, dmaConfig, switchConfig, lockConfig,
            coreConfig, previousDevice))) {
      loadPdiOp.emitError("Failed to generate reset operations");
      return failure();
    }
  }

  // Generate and insert configuration operations
  if (failed(xilinx::AIE::generateAndInsertConfigOps(builder, referencedDevice,
                                                     ""))) {
    loadPdiOp.emitError("Failed to generate configuration operations");
    return failure();
  }

  // Erase the original load_pdi operation
  loadPdiOp.erase();

  return success();
}

struct AIEExpandLoadPdiPass
    : public AIEExpandLoadPdiBase<AIEExpandLoadPdiPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
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

    // Collect all load_pdi operations in program order with their predecessors
    SmallVector<std::pair<NpuLoadPdiOp, AIE::DeviceOp>> loadPdiOps;
    AIE::DeviceOp previousDevice;

    module.walk([&](NpuLoadPdiOp loadPdiOp) {
      // Track the predecessor for this operation
      loadPdiOps.push_back({loadPdiOp, previousDevice});

      // Update previousDevice for the next load_pdi
      auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
      if (deviceRefAttr) {
        auto referencedDevice =
            module.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
        if (referencedDevice &&
            !referencedDevice.getSymName().starts_with("empty")) {
          previousDevice = referencedDevice;
        }
      }
    });

    // Transform load_pdi ops
    for (auto [loadPdiOp, prevDevice] : loadPdiOps) {
      if (failed(transformLoadPdi(loadPdiOp, prevDevice, module, dmaConfig,
                                  switchConfig, lockConfig, coreConfig))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
