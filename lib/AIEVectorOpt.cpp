//===- AIEVectorOpt.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "aie/AIEDialect.h"

#define DEBUG_TYPE "aie-vector-opt"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEVectorOptPass : public AIEVectorOptBase<AIEVectorOptPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    FuncOp f = getOperation();

    // Initial store->load forwarding
    vector::transferOpflowOpt(f);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalOp<vector::BroadcastOp>();
    // To start with, we're mainly interested in eliminating TransferRead ops
    // that can be converted to load + broadcast
    target.addDynamicallyLegalOp<vector::TransferReadOp>(
        [](vector::TransferReadOp op) { return false; });
    OwningRewritePatternList patterns(&getContext());
    vector::populateVectorTransferLoweringPatterns(patterns);
    vector::populateVectorMaskMaterializationPatterns(patterns, true);

    if (failed(applyPartialConversion(f, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<FuncOp>> xilinx::AIE::createAIEVectorOptPass() {
  return std::make_unique<AIEVectorOptPass>();
}