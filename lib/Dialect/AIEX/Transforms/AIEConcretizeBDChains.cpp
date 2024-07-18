//===- AIEConcretizeBDChains.cpp ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct AIEConcretizeBDChainsPass
    : AIEConcretizeBDChainsBase<AIEConcretizeBDChainsPass> {

  typedef DMAStartTask BDChainCallOp;

  LogicalResult inlineUsages() {
    InlinerConfig config;
    config.setMaxInliningIterations(1);

    Inliner::RunPipelineHelperTy runPipelineHelper =
        [&](Pass &pass, OpPassManager &pipeline, Operation *op) {
          return mlir::cast<AIEConcretizeBDChainsPass>(pass).runPipeline(
              pipeline, op);
        };

    CallGraph &cg = getAnalysis<CallGraph>();

    // The inliner should only be run on operations that define a symbol table,
    // as the callgraph will need to resolve references.
    Operation *op = getOperation();
    if (!op->hasTrait<OpTrait::SymbolTable>()) {
      op->emitOpError() << " was scheduled to run under the inliner, but does "
                           "not define a symbol table";
      return failure();
    }

    // We must inline at all call sites
    auto profitabilityCb = [=](const Inliner::ResolvedCall &call) {
      return true;
    };

    // Get an instance of the inliner.
    Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper,
                    config, profitabilityCb);

    // Run the inlining.
    if (failed(inliner.doInlining())) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {

    AIE::DeviceOp device = getOperation();

    // Wrap bd chains in DMAConfigureBDs regions before inlining
    device.walk([&](DMAStartTask start_op) {
      OpBuilder builder = OpBuilder(start_op);
      DMAConfigureBDs configure_op = builder.create<DMAConfigureBDs>(
          start_op.getLoc(), builder.getIndexType(), start_op.getTile());
      Block &b = configure_op.getBody().emplaceBlock();
      start_op->moveBefore(&b, b.end());
      builder.setInsertionPointAfter(start_op);
      builder.create<AIE::EndOp>(start_op.getLoc());
      builder.setInsertionPointAfter(configure_op);
      builder.create<DMAStartBDs>(start_op.getLoc(), configure_op.getResult(),
                                  start_op.getTile(), start_op.getDirection(),
                                  start_op.getChannel());
      // start_op->moveBefore(&b.back());
    });

    // Inline all usages of BD chains at their sites
    if (failed(inlineUsages())) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEConcretizeBDChainsPass() {
  return std::make_unique<AIEConcretizeBDChainsPass>();
}
