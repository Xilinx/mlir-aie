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

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESUBSTITUTESHIMDMAALLOCATIONS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct DMAConfigureTaskForOpPattern
    : public mlir::OpRewritePattern<DMAConfigureTaskForOp> {

  // Build-speed lever (byte-identical): the canonical path resolves each
  // task's shim-DMA allocation via AIE::ShimDMAAllocationOp::getForSymbol ->
  // device.lookupSymbol, a LINEAR symbol-table scan run once per
  // DMAConfigureTaskForOp. At B=128/L12 there are tens of thousands of these,
  // so the per-op O(allocations) scan makes the pass O(n^2) (measured
  // super-linear: L6 40s -> L12 156s = 3.86x for 2x layers). We instead build a
  // SymbolTable for the device ONCE and look up in O(1). Same fix as the
  // AIETargetNPU getDataWords symbol-cache. The pattern never erases/creates a
  // symbol (it rewrites task ops, not ShimDMAAllocationOps), so the prebuilt
  // table stays valid across the greedy run.
  const mlir::SymbolTable &symbolTable;

  DMAConfigureTaskForOpPattern(mlir::MLIRContext *ctx,
                               const mlir::SymbolTable &symbolTable)
      : OpRewritePattern<DMAConfigureTaskForOp>(ctx),
        symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(DMAConfigureTaskForOp op,
                                PatternRewriter &rewriter) const override {
    AIE::ShimDMAAllocationOp alloc_op =
        symbolTable.lookup<AIE::ShimDMAAllocationOp>(
            op.getAlloc().getRootReference());
    if (!alloc_op) {
      return op.emitOpError("no shim DMA allocation found for symbol");
    }

    AIE::TileOp tile = alloc_op.getTileOp();
    if (!tile) {
      return op.emitOpError(
          "shim DMA allocation must reference a valid TileOp");
    }

    DMAConfigureTaskOp new_op = DMAConfigureTaskOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), tile.getResult(),
        alloc_op.getChannelDir(), (int32_t)alloc_op.getChannelIndex(),
        op.getIssueToken(), op.getRepeatCount(),
        alloc_op.getPacket().value_or(nullptr));
    rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().begin());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIESubstituteShimDMAAllocationsPass
    : xilinx::AIEX::impl::AIESubstituteShimDMAAllocationsBase<
          AIESubstituteShimDMAAllocationsPass> {

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Build the device symbol table ONCE (O(1) per-task allocation lookup
    // instead of getForSymbol's per-task linear scan -> O(n^2)). Byte-identical.
    mlir::SymbolTable symbolTable(device);

    // Convert DMAConfigureTaskForOps that reference shim DMA allocations
    // to regular DMAConfigureTaskOps
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAConfigureTaskForOpPattern>(&getContext(), symbolTable);

    (void)applyPatternsGreedily(device, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIESubstituteShimDMAAllocationsPass() {
  return std::make_unique<AIESubstituteShimDMAAllocationsPass>();
}
