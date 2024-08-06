//===- AIEMaterializeBDChains.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <set>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAInlineBDChainPattern : OpConversionPattern<DMAStartBdChainOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAStartBdChainOp start_op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp device = start_op->getParentOfType<AIE::DeviceOp>();

    rewriter.setInsertionPointAfter(start_op);

    // Get referenced abstract BD chain
    AIE::BDChainOp chain_def = start_op.getBDChainOp();
    assert(chain_def);
    Region &source_region = chain_def.getBody();

    // Create BD op into which the result will be inlined
    DMAConfigureTaskOp configure_op = rewriter.create<DMAConfigureTaskOp>(
        start_op.getLoc(), rewriter.getIndexType(), start_op.getTile(),
        start_op.getDirection(), start_op.getChannel(),
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
    start_op.getResult().replaceAllUsesWith(configure_op.getResult());

    // Remove definition too if this was the only/last usage of it
    if (SymbolTable::symbolKnownUseEmpty(chain_def, device)) {
      chain_def.erase();
    }

    // Add a start BDs instruction
    rewriter.create<DMAStartTaskOp>(start_op.getLoc(),
                                    configure_op.getResult());

    // After fully inlining, remove the original instruction
    rewriter.eraseOp(start_op);

    return success();
  }
};

struct AIEMaterializeBDChainsPass
    : AIEMaterializeBDChainsBase<AIEMaterializeBDChainsPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    AIE::DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addIllegalOp<DMAStartBdChainOp>();
    RewritePatternSet patterns(ctx);
    patterns.insert<DMAInlineBDChainPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(device, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEMaterializeBDChainsPass() {
  return std::make_unique<AIEMaterializeBDChainsPass>();
}
