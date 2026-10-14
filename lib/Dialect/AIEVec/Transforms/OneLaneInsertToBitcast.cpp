//===- OneLaneInsertToBitcast.cpp - One-lane inserts as bitcasts *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Rewrites an llvm.insertelement of a scalar into a one-element vector as an
// llvm.bitcast, so that LLVM does not fold it with the extract of a lane into
// a shufflevector with a one-element result.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct OneLaneInsertToBitcastPattern
    : public OpRewritePattern<LLVM::InsertElementOp> {
  using OpRewritePattern<LLVM::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(insertOp.getType());
    if (!vecType || vecType.isScalable() || vecType.getNumElements() != 1)
      return failure();
    if (!matchPattern(insertOp.getPosition(), m_Zero()))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(insertOp, vecType,
                                                 insertOp.getValue());
    return success();
  }
};

struct OneLaneInsertToBitcastPass
    : public PassWrapper<OneLaneInsertToBitcastPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OneLaneInsertToBitcastPass)

  StringRef getArgument() const final {
    return "aievec-one-lane-insert-to-bitcast";
  }

  StringRef getDescription() const final {
    return "Build one-element LLVM vectors with a bitcast";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<OneLaneInsertToBitcastPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> xilinx::aievec::createOneLaneInsertToBitcastPass() {
  return std::make_unique<OneLaneInsertToBitcastPass>();
}
