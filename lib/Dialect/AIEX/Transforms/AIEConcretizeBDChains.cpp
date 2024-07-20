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

#include <set>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
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

    std::set<llvm::StringRef> concretized_syms = {};

    // Wrap bd chains in DMAConfigureBDs regions before inlining
    device.walk([&](DMAStartTask start_op) {
      OpBuilder builder = OpBuilder(start_op);
      DMAConfigureBDs configure_op = builder.create<DMAConfigureBDs>(
          start_op.getLoc(), builder.getIndexType(), 
          start_op.getTile(), start_op.getDirection(), start_op.getChannel(),
          start_op.getIssueToken(), start_op.getRepeatCount());
      Block &b = configure_op.getBody().emplaceBlock();
      start_op->moveBefore(&b, b.end());
      builder.setInsertionPointAfter(start_op);
      builder.create<AIE::EndOp>(start_op.getLoc());
      builder.setInsertionPointAfter(configure_op);
      builder.create<DMAStartBDs>(start_op.getLoc(), configure_op.getResult());
      concretized_syms.insert(start_op.getSymbol());
    });

    // Inline all usages of BD chains at their sites
    if (failed(inlineUsages())) {
      return signalPassFailure();
    }

    // Verify inlined basic blocks do form a chain reachable from the start;
    // Remove empty blocks
    WalkResult result = device.walk([&](DMAConfigureBDs configure_bds_op) {
      Region &body = configure_bds_op.getBody();
      for(auto it = body.begin(); it != body.end(); ++it) {
        Block &block = *it;
        auto ops_it = block.without_terminator();
        if(std::distance(ops_it.begin(), ops_it.end()) == 0) {
          block.erase();
          return WalkResult::advance();
        }
        if(block.hasNoPredecessors() && !block.isEntryBlock()) {
          auto error = block.getTerminator()->emitError("Block ending in this terminator does not form a chain with entry block.");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if(result.wasInterrupted()) {
      return signalPassFailure();
    }

    // If after concretizing no uses of the symbol are left, remove its definition
    for(auto it = concretized_syms.begin(); it != concretized_syms.end(); ++it) {
      llvm::StringRef sym = *it;
      Operation *def_op = SymbolTable::lookupSymbolIn(device, sym);
      if(!def_op) {
        continue;
      }
      if(SymbolTable::symbolKnownUseEmpty(def_op, device)) {
        assert(llvm::isa<AIE::BDChainOp>(def_op));
        def_op->erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEConcretizeBDChainsPass() {
  return std::make_unique<AIEConcretizeBDChainsPass>();
}
