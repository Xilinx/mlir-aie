//===- AIEVectorOpt.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-vector-opt"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEVectorOptPass : AIEVectorOptBase<AIEVectorOptPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    func::FuncOp f = getOperation();

    // Initial store->load forwarding
    IRRewriter rewriter(&getContext());
    vector::transferOpflowOpt(rewriter, f);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<vector::BroadcastOp>();
    // To start with, we're mainly interested in eliminating TransferRead ops
    // that can be converted to load + broadcast
    target.addDynamicallyLegalOp<vector::TransferReadOp>(
        [](vector::TransferReadOp op) { return false; });
    RewritePatternSet patterns(&getContext());
    vector::populateVectorTransferLoweringPatterns(patterns);
    vector::populateVectorMaskMaterializationPatterns(patterns, true);

    if (failed(applyPartialConversion(f, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> AIE::createAIEVectorOptPass() {
  return std::make_unique<AIEVectorOptPass>();
}