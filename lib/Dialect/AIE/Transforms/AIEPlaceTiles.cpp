//===- AIEPlaceTiles.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-place-tiles"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct ConvertLogicalTileToTile : OpConversionPattern<LogicalTileOp> {
  ConvertLogicalTileToTile(MLIRContext *context, DeviceOp &d,
                           PlacementAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

  LogicalResult
  matchAndRewrite(LogicalTileOp logicalTile, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get pre-computed placement from analysis
    auto placement = analyzer.getPlacement(logicalTile);
    if (!placement) {
      return logicalTile.emitError("no placement found for logical tile");
    }

    // handle merging multiple logical tiles to same physical tile
    TileOp tileOp =
        TileOp::getOrCreate(rewriter, device, placement->col, placement->row);

    // Copy allocation_scheme if present
    if (auto scheme = logicalTile.getAllocationScheme())
      tileOp.setAllocationScheme(scheme);

    // Replace all uses and erase logical tile
    rewriter.replaceOp(logicalTile, tileOp.getResult());
    return success();
  }

private:
  DeviceOp device;
  PlacementAnalysis &analyzer;
};

} // namespace

struct AIEPlaceTilesPass : AIEPlaceTilesBase<AIEPlaceTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    // Create placer
    std::shared_ptr<Placer> placer;
    if (clPlacerName == "sequential_placer") {
      placer = std::make_shared<SequentialPlacer>();
    } else {
      device.emitError() << "Unknown placer: " << clPlacerName;
      return signalPassFailure();
    }

    // Run placement analysis
    PlacementAnalysis analyzer(placer);
    if (failed(analyzer.runAnalysis(device)))
      return signalPassFailure();

    // Apply placement using conversion pattern
    ConversionTarget target(getContext());
    target.addLegalOp<TileOp>();
    target.addIllegalOp<LogicalTileOp>();

    // Mark all other AIE dialect operations as legal
    // They will have their operands automatically updated when LogicalTileOp ->
    // TileOp
    target.addLegalDialect<AIEDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLogicalTileToTile>(device.getContext(), device,
                                           analyzer);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}
