//===- AIESubstituteShimDMAAllocations.cpp -----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAConfigureTaskForOpPattern : RewritePattern {

  DMAConfigureTaskForOpPattern(MLIRContext *ctx)
      : RewritePattern(DMAConfigureTaskForOp::getOperationName(),
                       PatternBenefit(1), ctx) {}

  LogicalResult matchAndRewrite(Operation *op_any,
                                PatternRewriter &rewriter) const override {
    DMAConfigureTaskForOp op = llvm::dyn_cast<DMAConfigureTaskForOp>(op_any);
    if (!op) {
      return failure();
    }
    AIE::DeviceOp device = op->getParentOfType<AIE::DeviceOp>();

    AIE::ShimDMAAllocationOp alloc_op =
        AIE::ShimDMAAllocationOp::getForSymbol(device, op.getAlloc());
    if (!alloc_op) {
      return op.emitOpError("no shim DMA allocation found for symbol");
    }

    const int col = alloc_op.getCol();
    AIE::TileOp tile = AIE::TileOp::getOrCreate(rewriter, device, col, 0);
    DMAConfigureTaskOp new_op = rewriter.create<DMAConfigureTaskOp>(
        op.getLoc(), rewriter.getIndexType(), tile.getResult(),
        alloc_op.getChannelDir(), (int32_t)alloc_op.getChannelIndex(),
        op.getIssueToken(), op.getRepeatCount());
    rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().begin());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIESubstituteShimDMAAllocationsPass
    : AIESubstituteShimDMAAllocationsBase<AIESubstituteShimDMAAllocationsPass> {

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Convert DMAConfigureTaskForOps that reference shim DMA allocations
    // to regular DMAConfigureTaskOps
    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addIllegalOp<DMAConfigureTaskForOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAConfigureTaskForOpPattern>(&getContext());

    GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
    if (failed(applyPatternsAndFoldGreedily(device, std::move(patterns),
                                            rewriter_config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIESubstituteShimDMAAllocationsPass() {
  return std::make_unique<AIESubstituteShimDMAAllocationsPass>();
}
