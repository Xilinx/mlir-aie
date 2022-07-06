//===- AIEAssignLockIDs.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This pass aims to assign lockIDs to AIE.lock operations. The lockID is
// numbered from the most recent AIE.lock within the same tile. If the lockID
// exceeds 15 then the pass generates an error and terminates. AIE.lock
// operations for different tiles are numbered independently. If there are
// existing lock IDs, this pass is idempotent and only assign lock ids to locks
// without an ID.

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-assign-lock-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEAssignLockIDsPass
    : public AIEAssignLockIDsBase<AIEAssignLockIDsPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(m.getBody());

    // loop through locks
    // store lockID count in map with operation as key
    std::map<Operation *, std::pair<int, std::set<int>>> tileToLastID;
    for (auto lock : m.getOps<LockOp>()) {
      if (lock.lockID().hasValue()) {
        Operation *lock_tile = lock.tile().getDefiningOp();
        tileToLastID[lock_tile].first = 0;
        tileToLastID[lock_tile].second.insert(lock.getLockID());
      }
    }

    for (auto lock : m.getOps<LockOp>()) {
      Operation *lock_tile = lock.tile().getDefiningOp();

      if (!lock.lockID().hasValue()) {
        if (unique_tiles.find(lock_tile) == unique_tiles.end()) {
          // if not in map initial LockID = 0
          unique_tiles[lock_tile].first = 0;
        } else if (unique_tiles[lock_tile] < 15) {
          // if in map increment LockID
          unique_tiles[lock_tile] += 1;
        } else {
          lock->emitError() << "Exceeded the number of unique LockIDs";
          return;
        }

        lock->setAttr("lockID",
                      rewriter.getI32IntegerAttr(unique_tiles[lock_tile]));
      }
    }
  }
}
}
;

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}