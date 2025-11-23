//===- AIEExpandLoadPdi.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
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

#define DEBUG_TYPE "aie-expand-load-pdi"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct ExpandLoadPdiPattern : public OpRewritePattern<NpuLoadPdiOp> {
  using OpRewritePattern<NpuLoadPdiOp>::OpRewritePattern;

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
    auto emptyDevice = rewriter.create<AIE::DeviceOp>(
        loc, deviceType, rewriter.getStringAttr(emptyName));
    emptyDevice.getRegion().emplaceBlock();
    
    // Add empty cores in all columns to trigger column reset
    Block *deviceBlock = &emptyDevice.getRegion().front();
    
    //const auto &targetModel = referencedDevice.getTargetModel();
    //int numCols = targetModel.columns();
    //for (int col = 0; col < numCols; col++) {
    //  rewriter.setInsertionPointToEnd(deviceBlock);
    //  auto tile = rewriter.create<AIE::TileOp>(loc, col, 2);
    //  rewriter.setInsertionPointToEnd(deviceBlock);
    //  auto core = rewriter.create<AIE::CoreOp>(loc, tile.getResult());
    //  Region &coreRegion = core.getBody();
    //  Block *coreBlock = rewriter.createBlock(&coreRegion);
    //  rewriter.setInsertionPointToEnd(coreBlock);
    //  rewriter.create<AIE::EndOp>(loc);
    //}
    rewriter.setInsertionPointToEnd(deviceBlock);
    rewriter.create<AIE::EndOp>(loc);

    // Create new load_pdi operation referencing the empty device
    rewriter.setInsertionPoint(loadPdiOp);
    auto newLoadPdi = rewriter.create<NpuLoadPdiOp>(
        loadPdiOp.getLoc(), FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
        loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
        loadPdiOp.getAddressAttr());

    // Generate and insert reset operations after the empty load_pdi
    // if (failed(xilinx::AIE::generateAndInsertResetOps(
    //         referencedDevice, newLoadPdi.getOperation()))) {
    //   loadPdiOp.emitError("Failed to generate reset operations");
    //   return failure();
    // }

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

    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandLoadPdiPattern>(&getContext());

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
