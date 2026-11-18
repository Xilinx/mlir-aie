//===- AIEMaterializeBDChains.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEMATERIALIZEBDCHAINS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

// Concretizes `chain` with `args` as a dma_configure_task on `tile`'s
// channel, starts it, and replaces `start` with the task.
static void inlineBDChain(PatternRewriter &rewriter, Operation *start,
                          AIE::BDChainOp chain, ValueRange args, Value tile,
                          AIE::DMAChannelDir direction, int32_t channel,
                          bool issueToken, int32_t repeatCount,
                          FlatSymbolRefAttr alloc) {
  rewriter.setInsertionPointAfter(start);
  DMAConfigureTaskOp configure_op = DMAConfigureTaskOp::create(
      rewriter, start->getLoc(), rewriter.getIndexType(), tile, direction,
      channel, issueToken, repeatCount);
  configure_op.setAllocAttr(alloc);
  Region &source_region = chain.getBody();
  Region &target_region = configure_op.getBody();

  // Clone BD definition into usage site, replacing abstract SSA values with
  // concrete ones
  IRMapping arg_map;
  for (unsigned i = 0, n = source_region.getNumArguments(); i < n; i++) {
    BlockArgument arg = source_region.getArgument(i);
    Value val = args[i];
    assert(arg.getType() == val.getType());
    arg_map.map(arg, val);
  }
  source_region.cloneInto(&target_region, arg_map);

  rewriter.replaceAllUsesWith(start->getResult(0), configure_op.getResult());
  DMAStartTaskOp::create(rewriter, start->getLoc(), configure_op.getResult());
  rewriter.eraseOp(start);
}

struct DMAStartBdChainForOpPattern : RewritePattern {

  // Symbol-lookup cache used for speedier `ShimDMAAllocationOp` lookups
  const mlir::SymbolTable &symbolTable;

  DMAStartBdChainForOpPattern(MLIRContext *ctx,
                              const mlir::SymbolTable &symbolTable)
      : RewritePattern(DMAStartBdChainForOp::getOperationName(),
                       PatternBenefit(1), ctx),
        symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(Operation *op_any,
                                PatternRewriter &rewriter) const override {
    DMAStartBdChainForOp op = llvm::dyn_cast<DMAStartBdChainForOp>(op_any);
    if (!op) {
      return failure();
    }

    AIE::ShimDMAAllocationOp alloc_op =
        symbolTable.lookup<AIE::ShimDMAAllocationOp>(op.getAlloc());
    if (!alloc_op) {
      return op.emitOpError("no shim DMA allocation found for symbol");
    }

    AIE::TileOp tile = alloc_op.getTileOp();
    if (!tile) {
      return op.emitOpError(
          "shim DMA allocation must reference a valid TileOp");
    }

    auto chain = symbolTable.lookup<AIE::BDChainOp>(op.getSymbol());
    if (!chain) {
      return op.emitOpError("symbol does not reference valid BD chain");
    }

    inlineBDChain(rewriter, op, chain, op.getArgs(), tile.getResult(),
                  alloc_op.getChannelDir(), (int32_t)alloc_op.getChannelIndex(),
                  op.getIssueToken(), op.getRepeatCount(), op.getAllocAttr());
    return success();
  }
};

struct DMAInlineBDChainPattern : RewritePattern {

  DMAInlineBDChainPattern(MLIRContext *ctx)
      : RewritePattern(DMAStartBdChainOp::getOperationName(), PatternBenefit(1),
                       ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    DMAStartBdChainOp start_op = llvm::dyn_cast<DMAStartBdChainOp>(op);
    if (!start_op) { // Not a match.
      return failure();
    }
    AIE::BDChainOp chain_def = start_op.getBDChainOp();
    assert(chain_def);
    inlineBDChain(rewriter, start_op, chain_def, start_op.getArgs(),
                  start_op.getTile(), start_op.getDirection(),
                  start_op.getChannel(), start_op.getIssueToken(),
                  start_op.getRepeatCount(), FlatSymbolRefAttr());
    return success();
  }
};

struct AIEMaterializeBDChainsPass
    : xilinx::AIEX::impl::AIEMaterializeBDChainsBase<
          AIEMaterializeBDChainsPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    AIE::DeviceOp device = getOperation();
    GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
    rewriter_config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);

    mlir::SymbolTable symbolTable(device);

    RewritePatternSet patterns_0(ctx);
    patterns_0.insert<DMAStartBdChainForOpPattern>(ctx, symbolTable);
    DMAConfigureTaskOp::getCanonicalizationPatterns(patterns_0, ctx);
    if (failed(applyPatternsGreedily(device, std::move(patterns_0),
                                     rewriter_config))) {
      signalPassFailure();
    }

    RewritePatternSet patterns_1(ctx);
    patterns_1.insert<DMAInlineBDChainPattern>(ctx);
    rewriter_config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);
    DMAConfigureTaskOp::getCanonicalizationPatterns(patterns_1, ctx);
    if (failed(applyPatternsGreedily(device, std::move(patterns_1),
                                     rewriter_config))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEMaterializeBDChainsPass() {
  return std::make_unique<AIEMaterializeBDChainsPass>();
}
