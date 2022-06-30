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
    std::map<Operation *, int> tileToLastID;
    auto lockOpsRange = m.getOps<LockOp>();
    for (auto it = lockOpsRange.begin(); it != lockOpsRange.end(); it++) {
      Operation *lock_tile = (*it).tile().getDefiningOp();

      if (!(*it).lockID().hasValue()) {
        if (tileToLastID.find(lock_tile) == tileToLastID.end()) {
          // if not in map initial LockID = 0
          tileToLastID[lock_tile] = 0;
        } else if (tileToLastID[lock_tile] < 15) {
          // the next potential lockID
          int targetID = tileToLastID.at(lock_tile) + 1;

          // look ahead to check if targetID is taken, if so increment
          for (auto iit = it; iit != lockOpsRange.end(); iit++) {
            if ((*iit).lockID().hasValue() &&
                ((*iit).tile().getDefiningOp() == lock_tile &&
                 targetID == (*iit).getLockID()))
              targetID++;
          }

          // store lockID to map
          tileToLastID.at(lock_tile) = targetID;
        } else {
          (*it)->emitError() << "Exceeded the number of unique LockIDs";
          return;
        }
        (*it)->setAttr("lockID",
                       rewriter.getI32IntegerAttr(tileToLastID[lock_tile]));
      } else {
        (*it)->emitRemark() << "The Lock has an existing LockID\n";
      }

      // set LockID: overwrites existing LockID to maintain consistency
      // generate warning if Lock has an existing LockID and overwrite
      // if ((*it).lockID().hasValue())
      //   (*it)->emitWarning() << "The Lock has an existing LockID:
      //   Overwriting";
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}