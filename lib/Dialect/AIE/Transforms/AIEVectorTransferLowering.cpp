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

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEVECTORTRANSFERLOWERING
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-vector-transfer-opt"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIEVectorTransferLoweringPass
    : xilinx::AIE::impl::AIEVectorTransferLoweringBase<
          AIEVectorTransferLoweringPass> {
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
    DeviceOp deviceOp = getOperation();
    MLIRContext *context = &getContext();

    std::optional<unsigned> maxRank = std::nullopt;
    if (maxTransferRank != static_cast<unsigned>(-1))
      maxRank = maxTransferRank;

    RewritePatternSet patterns(context);
    vector::populateVectorTransferLoweringPatterns(patterns, maxRank);

    // Disable cross-region constant CSE to prevent the greedy rewriter from
    // hoisting arith.constant ops from inside aie.runtime_sequence (which has
    // IsolatedFromAbove) up to the aie.device scope. Without this, the default
    // cseConstants=true causes constants to be moved from the runtime_sequence
    // body into the device scope, making aie.core bodies reference device-scope
    // values. This breaks AIECoreToStandardPass which cannot clone the core
    // body into a func.func when the core references values defined outside it.
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);

    if (failed(applyPatternsGreedily(deviceOp, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEVectorTransferLoweringPass() {
  return std::make_unique<AIEVectorTransferLoweringPass>();
}
