//===- AIEVectorTransferLowering.cpp -----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aie-vector-transfer-opt"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIEVectorTransferLoweringPass
    : AIEVectorTransferLoweringBase<AIEVectorTransferLoweringPass> {
  AIEVectorTransferLoweringPass() = default;
  AIEVectorTransferLoweringPass(const AIEVectorTransferLoweringPass &pass)
      : AIEVectorTransferLoweringPass() {}
  AIEVectorTransferLoweringPass(unsigned maxTransferRank) {
    this->maxTransferRank = maxTransferRank;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    std::optional<unsigned> maxRank = std::nullopt;
    if (maxTransferRank != static_cast<unsigned>(-1))
      maxRank = maxTransferRank;
    {
      RewritePatternSet patterns(context);

      vector::populateVectorTransferLoweringPatterns(patterns, maxRank);

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
        signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
AIE::createAIEVectorTransferLoweringPass() {
  return std::make_unique<AIEVectorTransferLoweringPass>();
}
