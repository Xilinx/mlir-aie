//===- AIEMaterializeBDChains.cpp ---------------------------------*- C++
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

struct AIEMaterializeBDChainsPass
    : AIEMaterializeBDChainsBase<AIEMaterializeBDChainsPass> {

  WalkResult inlineUsage(AIE::DeviceOp device, DMAStartBdChainOp start_op) {
    OpBuilder builder = OpBuilder(start_op);

    // Get referenced abstract BD chain
    AIE::BDChainOp chain_def = start_op.getBDChainOp();
    assert(chain_def);
    Region &source_region = chain_def.getBody();

    // Create BD op into which the result will be inlined
    DMAConfigureTaskOp configure_op = builder.create<DMAConfigureTaskOp>(
        start_op.getLoc(), builder.getIndexType(), 
        start_op.getTile(), start_op.getDirection(), start_op.getChannel(),
        start_op.getIssueToken(), start_op.getRepeatCount());
    Region &target_region = configure_op.getBody();

    // Clone BD definition into usage site, replacing abstract SSA values with concrete ones
    IRMapping arg_map;
    ValueRange values = start_op.getConcreteArgs();
    for (unsigned i = 0, n = source_region.getNumArguments(); i < n; i++) {
      BlockArgument arg = source_region.getArgument(i);
      Value val = values[i];
      assert(arg.getType() == val.getType());
      arg_map.map(arg, val);
    }
    source_region.cloneInto(&target_region, arg_map);

    // Replace result of dma start task with result of bd chain configuration
    start_op.getResult().replaceAllUsesWith(configure_op.getResult());

    // Remove definition too if this was the only/last usage of it
    if(SymbolTable::symbolKnownUseEmpty(chain_def, device)) {
      chain_def.erase();
    }

    // Add a start BDs instruction
    builder.create<DMAStartTaskOp>(start_op.getLoc(), configure_op.getResult());

    // After fully inlining, remove the original instruction
    start_op.erase();

    return WalkResult::advance();
  }

  void runOnOperation() override {
    WalkResult r;

    AIE::DeviceOp device = getOperation();

    // Wrap bd chains in DMAConfigureTaskOp regions before inlining
    r = device.walk([&](DMAStartBdChainOp start_op) {
      return inlineUsage(device, start_op);
    });
    if(r.wasInterrupted()) {
      return signalPassFailure();
    }

    // Verify inlined basic blocks do form a chain reachable from the start;
    // Remove empty blocks
    r = device.walk([&](DMAConfigureTaskOp configure_task_op) {
      Region &body = configure_task_op.getBody();
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
    if(r.wasInterrupted()) {
      return signalPassFailure();
    }

  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEMaterializeBDChainsPass() {
  return std::make_unique<AIEMaterializeBDChainsPass>();
}
