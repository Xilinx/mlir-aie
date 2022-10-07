//===- CoreOp.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/Physical/CoreOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Rewrite/InlineFunction.h"

#include "aie/AIEDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::phy::physical;
using namespace xilinx::phy::rewrite;
using namespace xilinx::phy::transform::aie;

class CoreOpToAieLowering : public OpConversionPattern<CoreOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename CoreOp::Adaptor;

public:
  CoreOpToAieLowering(mlir::MLIRContext *context,
                      AIELoweringPatternSets *lowering)
      : OpConversionPattern<CoreOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(CoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = lowering->getTile(op);
    auto core = rewriter.replaceOpWithNewOp<xilinx::AIE::CoreOp>(op, tile);

    auto builder = OpBuilder::atBlockEnd(&core.getBody().emplaceBlock());
    auto callop = builder.create<func::CallOp>(
        builder.getUnknownLoc(), op.getCallee(), TypeRange(), op.operands());

    auto endop = builder.create<AIE::EndOp>(builder.getUnknownLoc());

    // failsafe: try to inline the call
    auto sym = callop.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    if (!sym)
      return success();

    auto func = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(callop, sym));
    if (!func)
      return success();

    Inliner inliner(rewriter.getContext());
    if (inlineCall(inliner, callop, func, func.getCallableRegion())
            .succeeded()) {
      // erase the call op if the inlining is done
      callop.erase();

      // if the endop block is not reachable, erase it as well
      auto *endblock = endop.getOperation()->getBlock();
      if (!endblock->isEntryBlock() && endblock->hasNoPredecessors()) {
        rewriter.eraseBlock(endblock);
      }
    }

    return success();
  }
};

void CoreOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {
  patterns.add<CoreOpToAieLowering>(patterns.getContext(), lowering);
}

void CoreOpLoweringPatternSet::populateTarget(mlir::ConversionTarget &target) {
  target.addLegalOp<func::CallOp>();
}
