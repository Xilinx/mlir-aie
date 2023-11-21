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
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

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

    // Map from tile to all its locks.
    DenseMap<TileOp, SmallVector<LockOp>> tileToLockOps;
    for (auto lockOp : device.getOps<LockOp>()) {
      auto tileOp = lockOp.getTileOp();
      auto iter = tileToLockOps.find(tileOp);
      if (iter == tileToLockOps.end()) {
        tileToLockOps.insert({tileOp, {lockOp}});
      } else {
        iter->second.push_back(lockOp);
      }
    }

    // For each tile, ensure that all its locks have unique and valid IDs.
    for (auto &&[tileOp, locks] : tileToLockOps) {

      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());

      // All lock IDs which are assigned to locks before this pass.
      DenseSet<int> idsPreAssigned;

      // All lock ops which do not have IDs before this pass.
      SmallVector<LockOp> idsNotAssigned;

      for (auto &&lockOp : locks) {
        if (lockOp.getLockID().has_value()) {
          uint32_t lockID = lockOp.getLockID().value();

          assert(lockID < locksPerTile &&
                 "This should be checked in the lock op verifier");

          // Duplicate preassigned lock ID:
          if (idsPreAssigned.find(lockID) != idsPreAssigned.end()) {
            tileOp->emitOpError("has multiple locks with ID ") << lockID << '.';
            return signalPassFailure();
          }
          idsPreAssigned.insert(lockID);
        } else {
          idsNotAssigned.push_back(lockOp);
        }
      }

      assert(idsPreAssigned.size() + idsNotAssigned.size() == locks.size());

      uint32_t nxtId = 0;
      for (auto &&lockOp : idsNotAssigned) {
        while (nxtId < locksPerTile &&
               (idsPreAssigned.find(nxtId) != idsPreAssigned.end())) {
          ++nxtId;
        }
        if (nxtId == locksPerTile) {
          tileOp->emitOpError("can have a maximum of ")
              << locksPerTile << " locks. No more available IDs.";
          return signalPassFailure();
        }
        lockOp->setAttr("lockID", rewriter.getI32IntegerAttr(nxtId));
        ++nxtId;
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}
