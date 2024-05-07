//===- AIEXToStandard.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

template <typename MyAIEXOp>
struct AIEXOpRemoval : OpConversionPattern<MyAIEXOp> {
  using OpConversionPattern<MyAIEXOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEXOp::Adaptor;
  ModuleOp &module;

  AIEXOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEXOp>(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(MyAIEXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEXToStandardPass : AIEXToStandardBase<AIEXToStandardPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AIEXOpRemoval<NpuDmaMemcpyNdOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuDmaWaitOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuShimTilePushQueueOp>>(m.getContext(),
                                                              m);
    removepatterns.add<AIEXOpRemoval<NpuWriteRTPOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWrite32Op>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuSyncOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWriteBdExShimTileOp>>(m.getContext(),
                                                              m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> AIEX::createAIEXToStandardPass() {
  return std::make_unique<AIEXToStandardPass>();
}
