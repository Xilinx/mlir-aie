//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

template <typename MyAIEXOp>
struct AIEXOpRemoval : public OpConversionPattern<MyAIEXOp> {
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

struct AIEXToStandardPass : public AIEXToStandardBase<AIEXToStandardPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    ConversionTarget target(getContext());
    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AIEXOpRemoval<AIEX::IpuDmaMemcpyNdOp>>(m.getContext(),
                                                              m);
    removepatterns.add<AIEXOpRemoval<AIEX::IpuShimTilePushQueueOp>>(
        m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<AIEX::IpuWriteRTPOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<AIEX::IpuWrite32Op>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<AIEX::IpuSyncOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<AIEX::IpuWriteBdExShimTileOp>>(
        m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEXCoreToStandardPass() {
  return std::make_unique<AIEXCoreToStandardPass>();
}
