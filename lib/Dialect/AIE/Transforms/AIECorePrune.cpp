//===- AIECorePrune.cpp ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-core-prune"

struct PruneTileOp : public OpRewritePattern<TileOp> {
  PruneTileOp(MLIRContext *context, int tileCol, int tileRow, bool pruneBuffers,
              bool pruneTiles)
      : OpRewritePattern<TileOp>(context), tileCol(tileCol), tileRow(tileRow),
        pruneBuffers(pruneBuffers), pruneTiles(pruneTiles) {}

  LogicalResult matchAndRewrite(TileOp tile,
                                PatternRewriter &rewriter) const override {
    if ((tile.getCol() == tileCol && tile.getRow() == tileRow) ||
        tile->hasAttr("pruned"))
      return failure();

    SetVector<Operation *> forwardSlice;
    getForwardSlice(tile.getResult(), &forwardSlice);
    SetVector<Operation *> erased;
    topologicalSort(forwardSlice);

    for (Operation *op : llvm::reverse(forwardSlice)) {
      if (isa<BufferOp>(op) && !pruneBuffers)
        continue;
      if (!erased.contains(op)) {
        rewriter.eraseOp(op);
        erased.insert(op);
      }
    }

    if (pruneTiles)
      rewriter.eraseOp(tile);
    else
      tile->setAttr("pruned", UnitAttr::get(getContext()));

    return success();
  }

  int tileCol;
  int tileRow;
  bool pruneBuffers;
  bool pruneTiles;
};

struct PruneUnusedExternalBuffers : public OpRewritePattern<ExternalBufferOp> {
  using OpRewritePattern<ExternalBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExternalBufferOp buffer,
                                PatternRewriter &rewriter) const override {
    if (!buffer.getResult().use_empty())
      return failure();

    rewriter.eraseOp(buffer);

    return success();
  }
};

struct AIECorePrunePass : AIECorePruneBase<AIECorePrunePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<PruneTileOp>(patterns.getContext(), tileCol, tileRow,
                              pruneBuffers, pruneTiles);
    patterns.add<PruneUnusedExternalBuffers>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> AIE::createAIECorePrunePass() {
  return std::make_unique<AIECorePrunePass>();
}
