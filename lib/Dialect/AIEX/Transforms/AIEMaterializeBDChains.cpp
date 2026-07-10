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

struct DMAStartBdChainForOpPattern : RewritePattern {

  // Build-speed lever (byte-identical): the canonical path resolves each
  // dma_start_bd_chain_for op's shim-DMA allocation via
  // AIE::ShimDMAAllocationOp::getForSymbol -> device.lookupSymbol, a LINEAR
  // symbol-table scan run once per DMAStartBdChainForOp. In large designs there
  // are many of these, so the per-op O(allocations) scan makes the pass O(n^2).
  // We instead build a SymbolTable for the device ONCE and look up in O(1). Same
  // fix as AIESubstituteShimDMAAllocations. The pattern never erases/creates a
  // symbol (it rewrites task ops, not ShimDMAAllocationOps), so the prebuilt
  // table stays valid across the greedy run.
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

    DMAStartBdChainOp new_op = DMAStartBdChainOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), op.getSymbol(),
        op.getArgs(), tile.getResult(), alloc_op.getChannelDir(),
        (int32_t)alloc_op.getChannelIndex(), op.getIssueToken(),
        op.getRepeatCount());
    rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
    rewriter.eraseOp(op);
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
    rewriter.setInsertionPointAfter(start_op);

    // Get referenced abstract BD chain
    AIE::BDChainOp chain_def = start_op.getBDChainOp();
    assert(chain_def);
    Region &source_region = chain_def.getBody();

    // Create BD op into which the result will be inlined
    DMAConfigureTaskOp configure_op = DMAConfigureTaskOp::create(
        rewriter, start_op.getLoc(), rewriter.getIndexType(),
        start_op.getTile(), start_op.getDirection(), start_op.getChannel(),
        start_op.getIssueToken(), start_op.getRepeatCount());
    Region &target_region = configure_op.getBody();

    // Clone BD definition into usage site, replacing abstract SSA values with
    // concrete ones
    IRMapping arg_map;
    ValueRange values = start_op.getArgs();
    for (unsigned i = 0, n = source_region.getNumArguments(); i < n; i++) {
      BlockArgument arg = source_region.getArgument(i);
      Value val = values[i];
      assert(arg.getType() == val.getType());
      arg_map.map(arg, val);
    }
    source_region.cloneInto(&target_region, arg_map);

    // Replace result of dma start task with result of bd chain configuration
    rewriter.replaceAllUsesWith(start_op.getResult(), configure_op.getResult());

    // Add a start BDs instruction
    DMAStartTaskOp::create(rewriter, start_op.getLoc(),
                           configure_op.getResult());

    // After fully inlining, remove the original instruction
    rewriter.eraseOp(start_op);

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
