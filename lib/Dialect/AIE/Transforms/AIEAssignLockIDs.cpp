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

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
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

    DeviceOp device = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(device.getBody());

    std::map<Operation *, std::pair<int, std::set<int>>> tileToLastID;

    // The first pass scans for and stores the existing lockIDs. This data is
    // stored in a map with the lock’s tile operation as the key, while the
    // value to the key is a pair with the current potential lockID and a set
    // that stores the currently assigned lockIDs.
    for (auto lock : device.getOps<LockOp>()) {
      if (lock.getLockID().has_value()) {
        Operation *lock_tile = lock.getTile().getDefiningOp();
        tileToLastID[lock_tile].first = 0;
        tileToLastID[lock_tile].second.insert(lock.getLockIDValue());
      }
    }

    // The second pass scans for locks with no lockIDs and assigns locks.
    for (auto lock : device.getOps<LockOp>()) {
      Operation *lock_tile = lock.getTile().getDefiningOp();
      if (!lock.getLockID().has_value()) {
        if (tileToLastID.find(lock_tile) == tileToLastID.end()) {
          // If the tile operation corresponding to the lock does not exist in
          // the data structure, initialize the lockID with 0 with an empty set.
          tileToLastID[lock_tile].first = 0;
        } else if (tileToLastID[lock_tile].first < 15) {
          // If the tile operation of the lock exists, the potential lockID is
          // checked with the set containing occupied lockIDs until a lockID
          // that is free is found.
          int potential_ID = tileToLastID[lock_tile].first;
          while (true) {
            if (tileToLastID[lock_tile].second.find(potential_ID) !=
                tileToLastID[lock_tile].second.end())
              potential_ID++;
            else
              break;
          }
          tileToLastID[lock_tile].first = potential_ID;
        } else {
          lock->emitError() << "Exceeded the number of unique LockIDs";
          return;
        }

        // The lockID is assigned and is stored in the set.
        lock->setAttr("lockID", rewriter.getI32IntegerAttr(
                                    tileToLastID[lock_tile].first));
        tileToLastID[lock_tile].second.insert(tileToLastID[lock_tile].first);
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}