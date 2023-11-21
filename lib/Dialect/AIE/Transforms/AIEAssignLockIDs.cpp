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
// exceeds the number of locks on the tile, the pass generates an error and
// terminates. AIE.lock operations for different tiles are numbered
// independently. If there are existing lock IDs, this pass is idempotent
// and only assigns lock IDs to locks without an ID.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "aie-assign-lock-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEAssignLockIDsPass : AIEAssignLockIDsBase<AIEAssignLockIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(device.getBody());

    struct TileLocks {
      DenseSet<int> assigned;
      SmallVector<LockOp> unassigned;
    };

    DenseMap<TileOp, TileLocks> locks;

    // Separate lock ops into assigned and unassigned ops, and store by tile.
    for (auto lockOp : device.getOps<LockOp>()) {

      auto tileOp = lockOp.getTileOp();
      bool isAssigned = lockOp.getLockID().has_value();

      // Append to set of assigned locks.
      if (isAssigned) {
        auto lockID = lockOp.getLockID().value();
        auto assignedLocksIter = locks.find(tileOp);
        if (assignedLocksIter == locks.end()) {
          locks.insert({tileOp, {{lockID}, {}}});
        } else {
          if (assignedLocksIter->second.assigned.find(lockID) !=
              assignedLocksIter->second.assigned.end()) {
            auto diag = lockOp->emitOpError("is assigned to the same lock (")
                        << lockID << ") as another op.";
            diag.attachNote(tileOp.getLoc())
                << "tile has lock ops assigned to same lock.";
            return signalPassFailure();
          }
          assignedLocksIter->second.assigned.insert(lockID);
        }
      }

      // Append to set of unassigned locks.
      else {
        auto unassignedLocksIter = locks.find(tileOp);
        if (unassignedLocksIter == locks.end()) {
          locks.insert({tileOp, {{}, {lockOp}}});
        } else {
          unassignedLocksIter->second.unassigned.push_back(lockOp);
        }
      }
    }

    // IR mutation: assign locks to all unassigned lock ops.
    for (auto &&[tileOp, tileLocks] : locks) {

      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());

      uint32_t nextID = 0;
      for (auto &&lockOp : tileLocks.unassigned) {
        while (nextID < locksPerTile &&
               (tileLocks.assigned.find(nextID) != tileLocks.assigned.end())) {
          ++nextID;
        }
        if (nextID == locksPerTile) {
          auto diag = lockOp->emitOpError("not allocated a lock.");
          diag.attachNote(tileOp.getLoc())
              << "tile has only " << locksPerTile << " locks available.";
          return signalPassFailure();
        }
        lockOp->setAttr("lockID", rewriter.getI32IntegerAttr(nextID));
        ++nextID;
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}
