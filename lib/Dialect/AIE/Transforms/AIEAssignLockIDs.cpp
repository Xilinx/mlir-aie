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

    // Map from tiles to all their lock ops (stored by ID) which have been
    // assigned to locks.
    DenseMap<TileOp, DenseSet<int>> tileToAssignedLocks;

    // Map from tiles to all lock ops which have not been assigned to locks.
    DenseMap<TileOp, SmallVector<LockOp>> tileToUnassignedLocks;

    // Separate lock ops into assigned and unassigned ops, and store by tile.
    for (auto lockOp : device.getOps<LockOp>()) {

      auto tileOp = lockOp.getTileOp();
      bool isAssigned = lockOp.getLockID().has_value();

      // Append to set of assigned locks.
      if (isAssigned) {
        auto lockID = lockOp.getLockID().value();
        auto assignedLocksIter = tileToAssignedLocks.find(tileOp);
        if (assignedLocksIter == tileToAssignedLocks.end()) {
          tileToAssignedLocks.insert({tileOp, {lockID}});
        } else {
          if (assignedLocksIter->second.find(lockID) !=
              assignedLocksIter->second.end()) {
            auto diag = lockOp->emitOpError("is assigned to the same lock (")
                        << lockID << ") as another op.";
            diag.attachNote(tileOp.getLoc())
                << "tile has lock ops assigned to same lock.";
            return signalPassFailure();
          }
          assignedLocksIter->second.insert(lockID);
        }
      }

      // Append to set of unassigned locks.
      else {
        auto unassignedLocksIter = tileToUnassignedLocks.find(tileOp);
        if (unassignedLocksIter == tileToUnassignedLocks.end()) {
          tileToUnassignedLocks.insert({tileOp, {lockOp}});
        } else {
          unassignedLocksIter->second.push_back(lockOp);
        }
      }
    }

    // IR mutation: assign locks to all unassigned lock ops.
    for (auto &&[tileOp, unassignedLocks] : tileToUnassignedLocks) {

      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());

      auto assignedLocksIter = tileToAssignedLocks.find(tileOp);

      // No locks have been assigned to this tile, so we don't do any collision
      // checking.
      if (assignedLocksIter == tileToAssignedLocks.end()) {
        if (unassignedLocks.size() >= locksPerTile) {
          auto diag = tileOp->emitOpError("has more lock ops (")
                      << unassignedLocks.size() << ") than locks ("
                      << locksPerTile << ").";
          return signalPassFailure();
        }
        for (auto lockOp : llvm::enumerate(unassignedLocks)) {
          lockOp.value()->setAttr("lockID",
                                  rewriter.getI32IntegerAttr(lockOp.index()));
        }
      }

      // Locks have been assigned to this tile, so we check for collisions when
      // assigning locks to lock ops.
      else {
        const auto &assignedLocks = assignedLocksIter->second;
        uint32_t nextID = 0;
        for (auto &&lockOp : unassignedLocks) {
          while (nextID < locksPerTile &&
                 (assignedLocks.find(nextID) != assignedLocks.end())) {
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
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}
