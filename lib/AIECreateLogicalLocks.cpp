//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This pass aims to assign lockIDs to AIE.lock operations. The lockID is
// numbered from the most recent AIE.lock within the same tile. If the lockID
// exceeds 15 then the pass generates an error and terminates. AIE.lock
// operations for different tiles are numbered independently. If there exists an
// existing lockID this pass overwrites the existing lockID generating a
// warning.

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

struct AIECreateLogicalLocksPass
    : public AIECreateLogicalLocksBase<AIECreateLogicalLocksPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(m.getBody());

    // loop through locks
    // store lockID count in map with operation as key
    std::map<Operation *, int> unique_tiles;
    for (auto lock : m.getOps<LockOp>()) {
      Operation *lock_tile = lock.tile().getDefiningOp();

      if (unique_tiles.find(lock_tile) == unique_tiles.end()) {
        // if not in map initial LockID = 0
        unique_tiles[lock_tile] = 0;
      } else if (unique_tiles[lock_tile] < 15) {
        // if in map increment LockID
        unique_tiles[lock_tile] += 1;
      } else {
        lock->emitError() << "Exceeded the number of unique LockIDs";
        return;
      }

      // set LockID: overwrites existing LockID to maintain consistency
      // generate warning if Lock has an existing LockID and overwrite
      if (lock.lockID().hasValue())
        lock->emitWarning() << "The Lock has an existing LockID: Overwriting";
      lock->setAttr("lockID",
                    rewriter.getI32IntegerAttr(unique_tiles[lock_tile]));
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECreateLogicalLocksPass() {
  return std::make_unique<AIECreateLogicalLocksPass>();
}
