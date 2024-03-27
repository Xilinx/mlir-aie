//===- AIELowerCascadeFlows.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-lower-cascade-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIELowerCascadeFlowsPass
    : AIELowerCascadeFlowsBase<AIELowerCascadeFlowsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {
    DeviceOp device = getOperation();
    std::shared_ptr<AIETargetModel> targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    std::set<TileOp> tilesWithCascadeFlow;
    DenseMap<TileOp, WireBundle> cascadeInputsPerTile;
    DenseMap<TileOp, WireBundle> cascadeOutputsPerTile;

    // identify cascade_flows and what ports they use on each tile
    for (auto cascadeFlow : device.getOps<CascadeFlowOp>()) {
      // for each cascade flow
      TileOp src = cascadeFlow.getSourceTileOp();
      TileOp dst = cascadeFlow.getDestTileOp();
      tilesWithCascadeFlow.insert(src);
      tilesWithCascadeFlow.insert(dst);

      if (targetModel->isSouth(src.getCol(), src.getRow(), dst.getCol(),
                               dst.getRow())) {
        cascadeInputsPerTile[dst] = WireBundle::North;
        cascadeOutputsPerTile[src] = WireBundle::South;
      } else if (targetModel->isEast(src.getCol(), src.getRow(), dst.getCol(),
                                     dst.getRow())) {
        cascadeInputsPerTile[dst] = WireBundle::West;
        cascadeOutputsPerTile[src] = WireBundle::East;
      } else {
        // TODO: remove when this pass supports routing
        cascadeFlow.emitOpError(
            "source tile must be to the North or West of the destination tile");
        return;
      }
    }

    // generate configure cascade ops
    for (TileOp tile : tilesWithCascadeFlow) {
      WireBundle inputDir;
      if (cascadeInputsPerTile.find(tile) != cascadeInputsPerTile.end()) {
        inputDir = cascadeInputsPerTile[tile];
      } else {
        inputDir = WireBundle::North;
      }
      WireBundle outputDir;
      if (cascadeOutputsPerTile.find(tile) != cascadeOutputsPerTile.end()) {
        outputDir = cascadeOutputsPerTile[tile];
      } else {
        outputDir = WireBundle::South;
      }
      builder.create<ConfigureCascadeOp>(builder.getUnknownLoc(), tile,
                                         static_cast<CascadeDir>(inputDir),
                                         static_cast<CascadeDir>(outputDir));
    }

    // erase CascadeFlowOps
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<CascadeFlowOp>(op))
        opsToErase.insert(op);
    });
    IRRewriter rewriter(&getContext());
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIELowerCascadeFlowsPass() {
  return std::make_unique<AIELowerCascadeFlowsPass>();
}
