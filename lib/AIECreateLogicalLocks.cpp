//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-logical-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIECreateLogicalLocksPass : public AIECreateLogicalLocksBase<AIECreateLogicalLocksPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    for (auto lock : m.getOps<LockOp>()) {
      LLVM_DEBUG(lock.dump());
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIECreateLogicalLocksPass() {
  return std::make_unique<AIECreateLogicalLocksPass>();
}
