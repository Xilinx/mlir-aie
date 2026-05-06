//===- AIEPlaceTiles.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEPLACETILES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-place-tiles"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct ConvertLogicalTileToTile : OpConversionPattern<LogicalTileOp> {
  ConvertLogicalTileToTile(MLIRContext *context, DeviceOp &d, Placer &p,
                           PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), placer(p) {}

  LogicalResult
  matchAndRewrite(LogicalTileOp logicalTile, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto placement = placer.getPlacement(logicalTile);
    if (!placement)
      return logicalTile.emitError("no placement found for logical tile");

    // Handle merging multiple logical tiles to same physical tile
    TileOp tileOp =
        TileOp::getOrCreate(rewriter, device, placement->col, placement->row);

    if (auto scheme = logicalTile.getAllocationScheme())
      tileOp.setAllocationScheme(scheme);

    rewriter.replaceOp(logicalTile, tileOp.getResult());
    return success();
  }

private:
  DeviceOp device;
  Placer &placer;
};

struct AIEPlaceTilesPass
    : xilinx::AIE::impl::AIEPlaceTilesBase<AIEPlaceTilesPass> {

  AIEPlaceTilesPass() = default;

  AIEPlaceTilesPass(const AIEPlaceTilesOptions &options) {
    clCoresPerCol = options.clCoresPerCol;
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    // Create placer
    std::shared_ptr<Placer> placer;
    switch (clPlacerType) {
    case PlacerType::SequentialPlacer: {
      std::optional<int> coresPerCol = std::nullopt;
      if (clCoresPerCol >= 0)
        coresPerCol = clCoresPerCol;
      placer = std::make_shared<SequentialPlacer>(coresPerCol);
      break;
    }
    }

    placer->initialize(device.getTargetModel());
    if (failed(placer->place(device)))
      return signalPassFailure();

    ConversionTarget target(getContext());
    target.addLegalOp<TileOp>();
    target.addIllegalOp<LogicalTileOp>();
    target.addLegalDialect<AIEDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLogicalTileToTile>(device.getContext(), device,
                                           *placer);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEPlaceTilesPass(const AIEPlaceTilesOptions &options) {
  return std::make_unique<AIEPlaceTilesPass>(options);
}
